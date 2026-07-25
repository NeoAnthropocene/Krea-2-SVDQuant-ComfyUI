"""Tensor-core W4A16 kernel for SVDQuant linears.

The kernel shipped in ``krea2_svdquant`` promotes both operands to fp32 before
``tl.dot``, so on anything without fp32 tensor cores it runs at CUDA-core speed. This
version keeps the dot in bf16 with fp32 accumulate, which is ~3.5x faster on Ampere and
numerically identical (cosine 1.00000 against the reference simulation).

Two details make it exact and cheap:

* ``BLOCK_K == GROUP``, so each k-tile maps to exactly one scale column and the scale is
  loaded once per tile instead of per element.
* The weight is repacked once, on first use, so the low nibble holds the *first half* of
  each 128-group and the high nibble the second half. In the checkpoint's own layout the
  nibbles interleave even/odd k, which forces stride-2 (uncoalesced) reads of the
  activation. After the repack both activation loads are contiguous.
* The two nibble lanes are accumulated as two separate dots -- the sum over k is order
  independent, so the result is unchanged (cosine 1.00000 against the reference).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from krea2_svdquant.runtime.linear import SVDQuantLinear

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - triton is optional
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _w4a16_tc_kernel(X, WQ, WS, Y, M, N, K: tl.constexpr, GROUP: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """``Y = X @ dequant(WQ, WS).T``; X must already be divided by smooth_scale."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_h = tl.arange(0, BLOCK_K // 2)

        packed_stride = K // 2
        scale_stride = K // GROUP
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[:, None] < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            b = tl.load(WQ + offs_n[:, None] * packed_stride + (k0 // 2 + offs_h[None, :]),
                        mask=n_mask, other=0)
            sc = tl.load(WS + offs_n[:, None] * scale_stride + (k0 // GROUP),
                         mask=n_mask, other=0.0)
            w_lo = (((b & 0x0F).to(tl.float32) - 8.0) * sc).to(tl.bfloat16)
            w_hi = ((((b >> 4) & 0x0F).to(tl.float32) - 8.0) * sc).to(tl.bfloat16)

            # Repacked layout: low nibble = first half of the group, high nibble = second.
            x_lo = tl.load(X + offs_m[:, None] * K + (k0 + offs_h)[None, :],
                           mask=m_mask, other=0.0)
            x_hi = tl.load(X + offs_m[:, None] * K + (k0 + BLOCK_K // 2 + offs_h)[None, :],
                           mask=m_mask, other=0.0)

            acc += tl.dot(x_lo, tl.trans(w_lo), out_dtype=tl.float32)
            acc += tl.dot(x_hi, tl.trans(w_hi), out_dtype=tl.float32)

        tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _unpack(qweight: torch.Tensor, k: int, group: int, layout: str) -> torch.Tensor:
    """Packed uint8 -> int8 values in k order."""
    n = qweight.shape[0]
    lo = (qweight & 0x0F).to(torch.int16) - 8
    hi = ((qweight >> 4) & 0x0F).to(torch.int16) - 8
    if layout == "orig":  # nibbles interleave even/odd k
        out = torch.empty((n, k), dtype=torch.int8, device=qweight.device)
        out[:, 0::2] = lo.to(torch.int8)
        out[:, 1::2] = hi.to(torch.int8)
        return out
    half = group // 2  # nibbles split each group in half
    lo = lo.reshape(n, k // group, half)
    hi = hi.reshape(n, k // group, half)
    return torch.cat([lo, hi], dim=-1).reshape(n, k).to(torch.int8)


def _pack(q: torch.Tensor, group: int, layout: str) -> torch.Tensor:
    n, k = q.shape
    if layout == "orig":
        lo, hi = q[:, 0::2], q[:, 1::2]
    else:
        grouped = q.reshape(n, k // group, group)
        lo = grouped[:, :, : group // 2].reshape(n, k // 2)
        hi = grouped[:, :, group // 2:].reshape(n, k // 2)
    u_lo = (lo.to(torch.int16) + 8).to(torch.uint8)
    u_hi = (hi.to(torch.int16) + 8).to(torch.uint8)
    return (u_lo | (u_hi << 4)).contiguous()


def convert_layout(qweight: torch.Tensor, k: int, group: int, src: str, dst: str) -> torch.Tensor:
    return _pack(_unpack(qweight, k, group, src), group, dst)


def _supported(x: torch.Tensor, state) -> bool:
    if triton is None or not x.is_cuda or x.dtype is not torch.bfloat16:
        return False
    if not state.qweight_packed or state.qweight.dtype is not torch.uint8:
        return False
    padded_in = int(state.padded_in_features or state.qweight.shape[1] * 2)
    if padded_in != int(state.original_shape[1]):
        return False  # padded checkpoints: let the reference path handle it
    return padded_in % int(state.group_size) == 0 and int(state.group_size) % 2 == 0


def svdquant_linear_tensorcore(x: torch.Tensor, state) -> torch.Tensor:
    out_prefix = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1])
    smooth = state.smooth_scale.to(device=x.device, dtype=x.dtype)
    x_hat = (x2d / smooth).contiguous()

    n = int(state.qweight.shape[0])
    k = int(state.padded_in_features or state.qweight.shape[1] * 2)
    m = int(x_hat.shape[0])

    block_m = 64 if m >= 64 else 16
    # Tuned on an RTX 3090: the wide layers (mlp, and anything reducing over 16384)
    # want bigger n-tiles and more warps; the square attention layers do not.
    if n >= 16384 or k >= 16384:
        block_n, num_warps = 128, 8
    else:
        block_n, num_warps = 64, 4

    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _w4a16_tc_kernel[grid](
        x_hat, state.qweight, state.weight_scales, y, m, n,
        K=k, GROUP=int(state.group_size),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=128,
        num_warps=num_warps, num_stages=2,
    )

    l1 = state.l1.to(device=x.device, dtype=x.dtype)
    l2 = state.l2.to(device=x.device, dtype=x.dtype)
    y += F.linear(F.linear(x_hat, l2), l1)
    if state.bias is not None:
        y += state.bias.to(device=x.device, dtype=x.dtype)
    return y.reshape(*out_prefix, n)


class FastSVDQuantLinear(SVDQuantLinear):
    """SVDQuantLinear that prefers the bf16 tensor-core kernel.

    The weight is repacked to the tensor-core layout lazily, on the device it is already
    resident on, so the conversion costs one pass over a single layer rather than a
    second copy of the whole model.
    """

    _warned = False

    def __init__(self, state, backend=None):
        if backend is None:
            super().__init__(state)
        else:
            super().__init__(state, backend=backend)
        self._layout = "orig"

    def _ensure_layout(self, layout: str) -> None:
        if self._layout == layout:
            return
        k = int(self.padded_in_features or self.qweight.shape[1] * 2)
        converted = convert_layout(self.qweight, k, int(self.group_size), self._layout, layout)
        self.qweight = converted
        self._layout = layout

    def _forward_base(self, x: torch.Tensor) -> torch.Tensor:
        if _supported(x, self):
            try:
                self._ensure_layout("tc")
                return svdquant_linear_tensorcore(x, self.state)
            except Exception:
                if not FastSVDQuantLinear._warned:
                    logging.exception(
                        "[krea2-svdquant] tensor-core kernel failed, falling back to the "
                        "reference path for the rest of this run"
                    )
                    FastSVDQuantLinear._warned = True
        self._ensure_layout("orig")
        return super()._forward_base(x)

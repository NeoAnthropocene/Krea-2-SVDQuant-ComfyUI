"""Route the two attention shapes sage-attention gets wrong back to stock attention.

**Float masks.** Sage's kernels take an ``attn_mask``, but a *float* (additive-bias) mask is
the one shape they handle badly:

* ``sageattn_qk_int8_pv_fp16_triton`` stages the mask tile through shared memory on
  top of the K/V pipeline. At head_dim 128 that is 139276 bytes against the 101376
  the SM offers on Ampere/Ada, so the launch dies with
  ``triton.runtime.errors.OutOfResources``.
* the CUDA kernels do launch, but the mask goes through the int8 path and the error
  against a bf16 reference grows ~50x versus the same call with no mask -- over 224
  layers that is a broken image, not a slightly noisy one.

Boolean masks are fine on both, and so is the no-mask case, which is every ordinary
text-to-image step. The only caller that hands Krea2 a float mask is
comfyui-krea2edit's ``ref_boost`` bias, so this guard costs nothing unless that
feature is on, and then it costs sage's speedup on the edit blocks alone.

**Short sequences.** ``sageattn_qk_int8_pv_fp16_cuda`` returns NaN for sequences shorter
than one K/V block, which is what ``auto`` resolves to on sm80/86/87 (``core.py``). Measured
on a 3090 at head_dim 128, same inputs every call, bf16 and fp16 alike::

    B=64  H=20  N=4..48   ->  7-10 of 10 calls non-finite
    B=64  H=20  N=63..128 ->  0 of 10
    B=592 H=20  N=12/32   ->  0-3 of 20, flips with the allocator state
    triton, every shape   ->  0

Krea2's ``txtfusion`` layerwise blocks attend over 12 tokens, so every step hits it, and one
NaN there poisons the whole latent: an all-black frame. The intermittency is why this looked
like a LoRA or quantization bug -- the same graph is fine on one model load and black on the
next. Sage has nothing to win on a 12-token attention anyway, so those calls go to stock
attention unconditionally rather than only under the kernels known to be broken.

Installed by the loader on the model it returns, and wrapped at sample time rather
than at load time because ComfyUI's Patch Sage Attention node may well run after us.
"""

from __future__ import annotations

import logging

import torch

import comfy.patcher_extension

_KEY = "krea2_sage_mask_guard"

# One K/V block. The measured boundary sits between 48 (broken) and 63 (clean), and 64 is
# the kernel's block size, so this is the block edge rather than a fudged constant.
_MIN_SEQ = 64

_warned: set[str] = set()


def _mask_of(args, kwargs):
    """The ``mask`` argument of ``optimized_attention*(q, k, v, heads, mask=...)``."""
    if "mask" in kwargs:
        return kwargs["mask"]
    return args[4] if len(args) > 4 else None


def _seq_len(q, kwargs):
    """Tokens per attention call, for both call shapes ComfyUI uses.

    ``skip_reshape`` means q already arrives as [B, H, N, D]; otherwise it is
    [B, N, H*D] and still has to be split by head.
    """
    if kwargs.get("skip_reshape"):
        return q.shape[-2]
    return q.shape[1] if q.ndim == 3 else q.shape[-2]


def _bail(reason, func, args, kwargs):
    if reason not in _warned:
        _warned.add(reason)
        logging.info("[krea2-svdquant] %s -> stock attention", reason)
    return func(*args, **kwargs)


def _guard(override):
    def guarded(func, *args, **kwargs):
        mask = _mask_of(args, kwargs)
        if mask is not None and torch.is_floating_point(mask):
            return _bail("float attention mask (sage mishandles additive-bias masks)",
                         func, args, kwargs)
        if _seq_len(args[0], kwargs) < _MIN_SEQ:
            return _bail("attention shorter than one K/V block (sage's CUDA kernel "
                         "returns NaN there)", func, args, kwargs)
        return override(func, *args, **kwargs)

    guarded.krea2_mask_guard = True
    return guarded


def _wrapper(executor, *args, **kwargs):
    transformer_options = kwargs.get("transformer_options")
    if transformer_options is None:
        for a in reversed(args):
            if isinstance(a, dict):
                transformer_options = a
                break
    if isinstance(transformer_options, dict):
        override = transformer_options.get("optimized_attention_override")
        if override is not None and not getattr(override, "krea2_mask_guard", False):
            transformer_options["optimized_attention_override"] = _guard(override)
    # __call__, not execute() -- execute() re-runs *this* wrapper (see WrapperExecutor).
    return executor(*args, **kwargs)


def install_mask_guard(patcher) -> None:
    transformer_options = patcher.model_options.setdefault("transformer_options", {})
    existing = transformer_options.get("wrappers", {}).get(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {})
    if _KEY in existing:
        return
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _KEY, _wrapper,
        transformer_options)

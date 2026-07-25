"""Quantize a BF16 Krea2 checkpoint into a ComfyUI-native quantized checkpoint.

ComfyUI already ships native kernels for these formats via comfy_kitchen, so the output
loads with a plain ``UNETLoader`` -- no custom node, and ordinary LoRA loaders work.

Two targets:

* ``int8``  -> ``int8_tensorwise`` with per-channel scales + convrot (W8A8).
               Natively accelerated on Ampere INT8 tensor cores.
* ``w4a4``  -> ``convrot_w4a4`` (W4A4), smaller and potentially faster, 4-bit quality.

Neither needs a calibration dataset: the convrot (group-wise Hadamard) rotation spreads
activation/weight outliers analytically, and activations are quantized by the kernel at
run time.

Only the 224 transformer-block linears are quantized, matching the layer set used by the
reference ``krea2_raw_int8_convrot`` checkpoint. Norms, modulation, the text-fusion stack
and the final layer stay in their original precision -- they are small and quantization
noise there is disproportionately visible.

    python quantize_krea2.py <bf16-model.safetensors> [--format int8|w4a4] [--out PATH]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

def _find_comfyui_root() -> str:
    """Locate the ComfyUI root so `comfy.*` imports work regardless of cwd.

    Preference order: $COMFYUI_PATH, then walking up from this file (this script
    lives in ComfyUI/custom_nodes/<pkg>/, so comfy/ is two levels up), then cwd.
    """
    env = os.environ.get("COMFYUI_PATH")
    if env and os.path.isdir(os.path.join(env, "comfy")):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(here, "..", ".."))
    if os.path.isdir(os.path.join(candidate, "comfy")):
        return candidate
    return "."


sys.path.insert(0, _find_comfyui_root())

from comfy.quant_ops import QUANT_ALGOS, get_layout_class  # noqa: E402

_QUANT_SUFFIXES = ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo",
                   "mlp.gate", "mlp.up", "mlp.down")


# Checkpoints ship either bare ("blocks.0...") or prefixed ("model.diffusion_model.blocks.0...").
_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "")

# Anything already quantized: we need the original high-precision weights to work from.
_QUANTIZED_DTYPES = ("F8_E4M3", "F8_E5M2", "I8", "U8")


def detect_prefix(keys) -> str:
    """Return the prefix the transformer blocks live under, or raise if not found."""
    for prefix in _PREFIXES:
        if any(k.startswith("{}blocks.".format(prefix)) for k in keys):
            return prefix
    raise SystemExit(
        "Could not find transformer blocks in this checkpoint. Expected keys like\n"
        "  blocks.0.attn.wq.weight  or  model.diffusion_model.blocks.0.attn.wq.weight\n"
        "This does not look like a Krea 2 diffusion model."
    )


def check_source_is_high_precision(handle, keys, prefix: str) -> None:
    """Refuse to re-quantize an already-quantized checkpoint.

    Quantizing FP8/INT8 weights down to 4 bits stacks one lossy step on another, and the
    SVD branch would be fitted to already-damaged weights. You need the BF16/FP16 release.
    """
    if any(k.endswith(".comfy_quant") for k in keys):
        raise SystemExit(
            "This checkpoint is already quantized (it carries `comfy_quant` markers).\n"
            "Re-quantizing it would stack a second lossy step on top of the first.\n"
            "Use the original BF16 (or FP16) release of the model as the source instead."
        )
    probe = "{}blocks.0.attn.wq.weight".format(prefix)
    if probe in keys:
        dtype = handle.get_slice(probe).get_dtype()
        if dtype in _QUANTIZED_DTYPES:
            raise SystemExit(
                "Source weights are {} - this checkpoint is already quantized.\n"
                "Use the original BF16 (or FP16) release of the model as the source "
                "instead.".format(dtype)
            )


def is_target(layer: str, prefix: str) -> bool:
    """Only the transformer blocks; txtfusion and friends stay high precision."""
    if not layer.startswith("{}blocks.".format(prefix)):
        return False
    return any(layer.endswith(s) for s in _QUANT_SUFFIXES)


def svd_lowrank(weight: torch.Tensor, rank: int, oversample: int = 16, niter: int = 2):
    """Randomized truncated SVD: return (L1 [out, rank], L2 [rank, in]) with W ~ L1 @ L2.

    Randomized rather than exact because these are 6144x16384 matrices and only the top
    few dozen singular directions matter here; `oversample` extra probe columns plus a
    couple of power iterations recover them to well past the precision 4-bit quantization
    cares about.
    """
    w = weight.float()
    min_dim = min(w.shape)
    rank = max(1, min(int(rank), min_dim))
    q = min(rank + max(0, int(oversample)), min_dim)

    if min_dim <= max(q, 32):
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        u_r, s_r, vh_r = u[:, :rank], s[:rank], vh[:rank, :]
    else:
        u, s, v = torch.svd_lowrank(w, q=q, niter=max(0, int(niter)))
        u_r, s_r, vh_r = u[:, :rank], s[:rank], v[:, :rank].transpose(-2, -1)

    return (u_r * s_r.unsqueeze(0)).contiguous(), vh_r.contiguous()


def svdquant_split(weight: torch.Tensor, rank: int):
    """SVDQuant ordering: pull a low-rank bf16 branch out of W, quantize the residual.

    The low-rank branch absorbs the outlier-heavy directions, so the part that has to
    survive 4 bits is better conditioned.
    """
    l1, l2 = svd_lowrank(weight.float(), rank, oversample=16, niter=2)
    l1 = l1.to(torch.bfloat16)
    l2 = l2.to(torch.bfloat16)
    residual = (weight.float() - (l1.float() @ l2.float())).to(torch.bfloat16)
    return residual, l1, l2


def quantize_weight(weight: torch.Tensor, fmt: str, groupsize: int):
    layout = get_layout_class(QUANT_ALGOS[fmt]["comfy_tensor_layout"])
    if fmt == "int8_tensorwise":
        qdata, params = layout.quantize(
            weight, is_weight=True, per_channel=True,
            convrot=True, convrot_groupsize=groupsize,
        )
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": groupsize}
        scales = {"weight_scale": params.scale}
    elif fmt == "convrot_w4a4":
        qdata, params = layout.quantize(weight, convrot_groupsize=groupsize)
        conf = {"format": "convrot_w4a4", "convrot_groupsize": groupsize,
                "linear_dtype": getattr(params, "linear_dtype", "int4")}
        scales = {"weight_scale": params.scale}
    else:
        raise ValueError(fmt)
    return qdata, scales, conf


def conf_tensor(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def convert(src: str, dst: str, fmt: str, groupsize: int, device: str = "cuda", rank: int = 0):
    out: dict[str, torch.Tensor] = {}
    quantized = 0
    kept = 0
    t0 = time.time()

    with safe_open(src, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        prefix = detect_prefix(keys)
        check_source_is_high_precision(handle, keys, prefix)
        for i, key in enumerate(keys):
            if key.endswith(".weight"):
                layer = key[: -len(".weight")]
                if is_target(layer, prefix):
                    w = handle.get_tensor(key).to(device=device, dtype=torch.bfloat16)
                    if rank > 0:
                        w, l1, l2 = svdquant_split(w, rank)
                        out["{}.svdq_l1".format(layer)] = l1.cpu()
                        out["{}.svdq_l2".format(layer)] = l2.cpu()
                        del l1, l2
                    qdata, scales, conf = quantize_weight(w, fmt, groupsize)
                    out[key] = qdata.cpu()
                    for name, value in scales.items():
                        out["{}.{}".format(layer, name)] = value.cpu()
                    out["{}.comfy_quant".format(layer)] = conf_tensor(conf)
                    del w, qdata, scales
                    quantized += 1
                    if quantized % 32 == 0:
                        torch.cuda.empty_cache()
                        print(f"  [{i + 1}/{len(keys)}] quantized {quantized} layers "
                              f"({time.time() - t0:.0f}s)", flush=True)
                    continue
            out[key] = handle.get_tensor(key)
            kept += 1

    gc.collect()
    torch.cuda.empty_cache()
    print(f"quantized {quantized} layers, kept {kept} tensors; writing {dst} ...", flush=True)
    save_file(out, dst)
    size = os.path.getsize(dst) / 1024 ** 3
    print(f"done in {time.time() - t0:.0f}s -> {dst}  ({size:.2f} GB, "
          f"source {os.path.getsize(src) / 1024 ** 3:.2f} GB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--format", choices=["int8", "w4a4", "svdq"], default="int8",
                    help="svdq = w4a4 residual + SVDQuant low-rank bf16 branch")
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--rank", type=int, default=64, help="low-rank size, svdq only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    fmt = "int8_tensorwise" if args.format == "int8" else "convrot_w4a4"
    if fmt not in QUANT_ALGOS:
        raise SystemExit(f"{fmt} is not available in this ComfyUI build")
    rank = args.rank if args.format == "svdq" else 0

    out = args.out
    if out is None:
        stem = os.path.splitext(os.path.basename(args.src))[0]
        suffix = f"svdq_r{rank}" if rank else f"{args.format}_convrot"
        out = os.path.join(os.path.dirname(args.src), f"{stem}_{suffix}.safetensors")
    convert(args.src, out, fmt, args.groupsize, args.device, rank)


if __name__ == "__main__":
    main()

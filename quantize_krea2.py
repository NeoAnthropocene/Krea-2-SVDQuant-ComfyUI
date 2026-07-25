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


def is_target(layer: str) -> bool:
    """Only the transformer blocks; txtfusion and friends stay high precision."""
    return layer.startswith("blocks.") and any(layer.endswith(s) for s in _QUANT_SUFFIXES)


def svdquant_split(weight: torch.Tensor, rank: int):
    """SVDQuant ordering: pull a low-rank bf16 branch out of W, quantize the residual.

    The low-rank branch absorbs the outlier-heavy directions, so the part that has to
    survive 4 bits is better conditioned.
    """
    from krea2_svdquant.quant.svd import svd_lowrank

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
        for i, key in enumerate(keys):
            if key.endswith(".weight"):
                layer = key[: -len(".weight")]
                if is_target(layer):
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

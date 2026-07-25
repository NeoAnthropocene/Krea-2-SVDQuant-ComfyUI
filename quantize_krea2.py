"""Quantize a BF16 Krea2 checkpoint into a ComfyUI-native quantized checkpoint.

ComfyUI already ships native kernels for these formats via comfy_kitchen, so the output
loads with a plain ``UNETLoader`` -- no custom node, and ordinary LoRA loaders work.

Four targets:

* ``int8``  -> ``int8_tensorwise`` with per-channel scales + convrot (W8A8).
               Natively accelerated on Ampere INT8 tensor cores.
* ``w4a4``  -> ``convrot_w4a4`` (W4A4), smaller and potentially faster, 4-bit quality.
* ``svdq``  -> the same W4A4 residual plus an SVDQuant low-rank bf16 branch, which
               absorbs the outlier directions 4 bits handle worst. Needs this repo's
               loader node; a plain ``UNETLoader`` cannot see the branch.
* ``fp8``   -> ``float8_e4m3fn``, a single per-tensor scale, no convrot, no calibration.
               Lightest touch of the four -- best fidelity, smallest speedup, and the
               only one that needs no rotation or outlier handling because 8-bit float
               already has enough dynamic range for these weights.

None of them need a calibration dataset: int8/w4a4/svdq spread outliers analytically via
the convrot (group-wise Hadamard) rotation, and activations are quantized by the kernel at
run time; fp8 just needs one abs-max scale per tensor.

Only the 224 transformer-block linears are quantized, matching the layer set used by the
reference ``krea2_raw_int8_convrot`` checkpoint. Norms, modulation, the text-fusion stack
and the final layer stay in their original precision -- they are small and quantization
noise there is disproportionately visible.

The layer selection is variant-agnostic: Krea 2 Turbo and the base (non-distilled)
release share the same block naming, so the same command converts either. What differs is
how you sample afterwards, not how you quantize -- pass ``--variant`` so the output is
named accordingly and the file records which one it came from.

    python quantize_krea2.py <bf16-model.safetensors> [--format int8|w4a4|svdq|fp8]
                             [--rank 64] [--variant turbo|base] [--out PATH]
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

__version__ = "0.2.0"

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

# Formats we know how to reconstruct back to BF16 from disk alone. FP8 storage is one
# byte per element with no packing or pre-rotation, so `qdata * scale` recovers the
# original tensor exactly as it was cast -- no architecture knowledge needed beyond
# what's already in the file. INT8/W4A4 sources pack multiple values per byte and are
# rotated (convrot) before quantization; unpacking those correctly needs the exact
# in/out feature counts from the live nn.Linear, which isn't recoverable from the
# checkpoint alone, so those are rejected instead.
_DEQUANTIZABLE_FORMATS = ("float8_e4m3fn", "float8_e5m2")
_UNSCALED_FP8_DTYPES = ("F8_E4M3", "F8_E5M2")


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


def check_requantizable(handle, keys, prefix: str) -> None:
    """Make sure every target layer that's already quantized is something we can
    dequantize back to BF16 (see `_DEQUANTIZABLE_FORMATS`). Anything else -- most
    importantly INT8/W4A4 -- needs the original BF16 (or FP16) release instead.
    """
    for key in keys:
        if not key.endswith(".comfy_quant"):
            continue
        layer = key[: -len(".comfy_quant")]
        if not is_target(layer, prefix):
            continue
        conf = json.loads(bytes(handle.get_tensor(key).tolist()))
        fmt = conf.get("format")
        if fmt not in _DEQUANTIZABLE_FORMATS:
            raise SystemExit(
                "Layer {} is already quantized as '{}'. Only FP8-quantized layers can be "
                "automatically reconstructed and re-quantized; for INT8/W4A4 sources, use "
                "the original BF16 (or FP16) release of the model instead.".format(layer, fmt)
            )


def dequantize_target_weight(handle, layer: str, device: str) -> torch.Tensor:
    """Load a target layer's weight as BF16, dequantizing it first if it's FP8.

    Scaled FP8 (has a `comfy_quant` marker + `weight_scale`): ``qdata.float() * scale``.
    Unscaled FP8 (a bare ``.to(float8_e4m3fn)`` cast, no marker): ``qdata`` as-is.
    Anything else is already ruled out by `check_requantizable`.
    """
    weight = handle.get_tensor("{}.weight".format(layer)).to(device=device)
    conf_key = "{}.comfy_quant".format(layer)
    if conf_key in handle.keys():
        scale = handle.get_tensor("{}.weight_scale".format(layer)).to(device=device).float()
        return (weight.to(torch.float32) * scale).to(torch.bfloat16)
    return weight.to(torch.bfloat16)


def is_target(layer: str, prefix: str) -> bool:
    """Only the transformer blocks; txtfusion and friends stay high precision."""
    if not layer.startswith("{}blocks.".format(prefix)):
        return False
    return any(layer.endswith(s) for s in _QUANT_SUFFIXES)


def leaf_name(layer: str, prefix: str) -> str | None:
    """``blocks.7.attn.wq`` -> ``attn.wq``. None if the layer is not inside a block.

    Only used to make a "nothing matched" failure self-diagnosing: a Krea 2 variant that
    keeps the ``blocks.`` container but renames its leaves would otherwise quantize zero
    layers and still write a full-size file.
    """
    head = "{}blocks.".format(prefix)
    if not layer.startswith(head):
        return None
    rest = layer[len(head):]
    return rest.split(".", 1)[1] if "." in rest else None


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


def svdquant_split(weight: torch.Tensor, rank: int, fmt: str, groupsize: int,
                   refine_iters: int = 100):
    """SVDQuant ordering: pull a low-rank bf16 branch out of W, quantize the residual.

    The low-rank branch absorbs the outlier-heavy directions, so the part that has to
    survive 4 bits is better conditioned.

    A single SVD of W is only the first guess: it picks the directions that are largest
    in W, which is not the same as the directions the quantizer handles worst. So this
    iterates the way deepcompressor does -- re-fit the branch to whatever the quantizer
    is *currently* getting wrong, requantize, repeat -- and keeps the best result. Pass
    ``refine_iters=0`` for the plain single-shot split.

    Iteration one is exactly that single-shot split and the best is kept, so refining
    can only match or beat it. Measured on Krea2 Turbo, rank 64: ~10% less
    reconstruction error, every layer improving.

    The objective is weight reconstruction error, which needs no calibration data.
    That is the true output error under the assumption that the input covariance is
    identity -- and spreading outliers with the convrot rotation is what makes that
    assumption a reasonable one. Closing the remaining gap to deepcompressor means
    measuring the real covariance, which is what makes their conversion take hours.

    Returns None for a degenerate weight (all-zero, or one that makes the error metric
    non-finite); the caller quantizes those without a branch rather than aborting.
    """
    w = weight.float()
    w_norm = torch.linalg.matrix_norm(w).item()
    if not math.isfinite(w_norm) or w_norm == 0.0:
        return None
    qw = torch.zeros((), device=w.device, dtype=torch.float32)

    best = None
    best_err = float("inf")
    for _ in range(max(1, refine_iters)):
        l1, l2 = svd_lowrank(w - qw, rank, oversample=16, niter=2)
        l1 = l1.to(torch.bfloat16)
        l2 = l2.to(torch.bfloat16)
        lw = l1.float() @ l2.float()
        residual = (w - lw).to(torch.bfloat16)

        qdata, params, layout, _ = _quantize_raw(residual, fmt, groupsize)
        qw = layout.dequantize(qdata, params).float()
        del qdata, params

        err = (torch.linalg.matrix_norm(w - (lw + qw)) / w_norm).item()
        del lw
        # NaN fails every comparison, so without this the loop would neither break nor
        # ever update `best` -- it would burn all `refine_iters` and return the last
        # split rather than the best one.
        if not math.isfinite(err):
            break
        if best is not None and err >= best_err - 1e-6:
            break  # refinement has stopped paying off
        best_err, best = err, (residual, l1, l2)

    return best


def _quantize_raw(weight: torch.Tensor, fmt: str, groupsize: int):
    """Quantize to `fmt`, handing back the layout and params too.

    Split out from `quantize_weight` so the refinement loop can round-trip a weight
    (quantize then dequantize) to see what the quantizer is actually getting wrong.
    """
    layout = get_layout_class(QUANT_ALGOS[fmt]["comfy_tensor_layout"])
    if fmt == "int8_tensorwise":
        qdata, params = layout.quantize(
            weight, is_weight=True, per_channel=True,
            convrot=True, convrot_groupsize=groupsize,
        )
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": groupsize}
    elif fmt == "convrot_w4a4":
        qdata, params = layout.quantize(weight, convrot_groupsize=groupsize)
        conf = {"format": "convrot_w4a4", "convrot_groupsize": groupsize,
                "linear_dtype": getattr(params, "linear_dtype", "int4")}
    elif fmt == "float8_e4m3fn":
        qdata, params = layout.quantize(weight, scale="recalculate")
        conf = {"format": "float8_e4m3fn"}
    else:
        raise ValueError(fmt)
    return qdata, params, layout, conf


def quantize_weight(weight: torch.Tensor, fmt: str, groupsize: int):
    qdata, params, _, conf = _quantize_raw(weight, fmt, groupsize)
    return qdata, {"weight_scale": params.scale}, conf


def conf_tensor(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def convert(src: str, dst: str, fmt: str, groupsize: int, device: str = "cuda", rank: int = 0,
            refine_iters: int = 100, variant: str = "unknown"):
    out: dict[str, torch.Tensor] = {}
    quantized = 0
    kept = 0
    branched = 0
    observed_leaves: set[str] = set()
    t0 = time.time()

    with safe_open(src, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        prefix = detect_prefix(keys)
        check_requantizable(handle, keys, prefix)

        # Companion keys (old scales/markers) for target layers get regenerated fresh
        # when we process the ".weight" key below -- skip the stale copies on disk,
        # otherwise they'd overwrite our new ones later in iteration order.
        stale_companions = set()
        for key in keys:
            if key.endswith(".weight"):
                layer = key[: -len(".weight")]
                if is_target(layer, prefix):
                    for suffix in ("weight_scale", "weight_scale_2", "input_scale", "comfy_quant"):
                        stale_companions.add("{}.{}".format(layer, suffix))

        for i, key in enumerate(keys):
            if key.endswith(".weight"):
                layer = key[: -len(".weight")]
                leaf = leaf_name(layer, prefix)
                if leaf is not None:
                    observed_leaves.add(leaf)
                if is_target(layer, prefix):
                    w = dequantize_target_weight(handle, layer, device)
                    if rank > 0:
                        split = svdquant_split(w, rank, fmt, groupsize, refine_iters)
                        if split is None:
                            print("  warning: {} is degenerate (zero or non-finite); "
                                  "quantizing it without a low-rank branch".format(layer),
                                  flush=True)
                        else:
                            w, l1, l2 = split
                            out["{}.svdq_l1".format(layer)] = l1.cpu()
                            out["{}.svdq_l2".format(layer)] = l2.cpu()
                            branched += 1
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
            if key in stale_companions:
                continue
            out[key] = handle.get_tensor(key)
            kept += 1

    gc.collect()
    torch.cuda.empty_cache()

    if quantized == 0:
        raise SystemExit(
            "Found the transformer blocks but quantized nothing: no layer under "
            "'{}blocks.' ends in one of the expected leaf names.\n"
            "  expected: {}\n"
            "  observed: {}\n"
            "This is a Krea 2 variant with different layer naming; quantize_krea2.py "
            "would otherwise have written a full-size file that does nothing."
            .format(prefix, ", ".join(sorted(_QUANT_SUFFIXES)),
                    ", ".join(sorted(observed_leaves)) or "none")
        )

    created = len(out) - kept - quantized
    factors = "" if not rank else " + {} low-rank factors".format(branched * 2)
    print(f"quantized {quantized} layers; {kept} tensors passed through; "
          f"{created} tensors created (qdata/scale/marker{factors}); "
          f"{len(out)} tensors total; writing {dst} ...", flush=True)

    # Recorded so the loader (and anyone inspecting the file) can tell what produced it.
    # safetensors metadata is str -> str only. The loader treats all of this as optional:
    # checkpoints published before this existed still load, with the rank recovered from
    # the svdq_l1 shape as before.
    metadata = {
        "krea2_svdquant_tool_version": __version__,
        "krea2_svdquant_format": fmt,
        "krea2_svdquant_rank": str(rank),
        "krea2_svdquant_groupsize": str(groupsize),
        "krea2_svdquant_refine_iters": str(refine_iters if rank else 0),
        "krea2_svdquant_variant": variant,
        "krea2_svdquant_quantized_layers": str(quantized),
        "krea2_svdquant_branched_layers": str(branched),
        "krea2_svdquant_source_name": os.path.basename(src),
        "krea2_svdquant_source_bytes": str(os.path.getsize(src)),
    }
    save_file(out, dst, metadata=metadata)
    size = os.path.getsize(dst) / 1024 ** 3
    print(f"done in {time.time() - t0:.0f}s -> {dst}  ({size:.2f} GB, "
          f"source {os.path.getsize(src) / 1024 ** 3:.2f} GB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--format", choices=["int8", "w4a4", "svdq", "fp8"], default="int8",
                    help="svdq = w4a4 residual + SVDQuant low-rank bf16 branch; "
                         "fp8 = float8_e4m3fn, no convrot, no low-rank branch")
    ap.add_argument("--groupsize", type=int, default=256, help="unused for fp8")
    ap.add_argument("--rank", type=int, default=64, help="low-rank size, svdq only")
    ap.add_argument("--refine-iters", type=int, default=100,
                    help="svdq only: refine the low-rank branch against the quantization "
                         "error, keeping the best (0 = plain single-shot SVD, much faster "
                         "but ~10%% more reconstruction error)")
    ap.add_argument("--variant", choices=["turbo", "base", "unknown"], default="unknown",
                    help="which Krea 2 release this is. Only affects the output filename "
                         "and the recorded metadata -- quantization is identical for both, "
                         "the difference is the sampler settings you run afterwards")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    fmt = {
        "int8": "int8_tensorwise",
        "w4a4": "convrot_w4a4",
        "svdq": "convrot_w4a4",
        "fp8": "float8_e4m3fn",
    }[args.format]
    if fmt not in QUANT_ALGOS:
        raise SystemExit(f"{fmt} is not available in this ComfyUI build")
    # Silently zeroing the rank here used to make `--format w4a4 --rank 128` look like it
    # had done something it had not.
    if args.format != "svdq" and args.rank != ap.get_default("rank"):
        raise SystemExit(
            "--rank only applies to --format svdq (you asked for --format {}). "
            "The low-rank branch is what distinguishes svdq from plain w4a4."
            .format(args.format))
    rank = args.rank if args.format == "svdq" else 0

    out = args.out
    if out is None:
        stem = os.path.splitext(os.path.basename(args.src))[0]
        if args.variant != "unknown":
            stem = "Krea2-{}".format(args.variant.capitalize())
        elif stem.lower() in ("raw", "model", "diffusion_pytorch_model", "turbo"):
            print("note: '{}' is a generic filename. Pass --variant turbo|base (or --out) "
                  "to get a checkpoint name you will still recognise next month."
                  .format(stem), flush=True)
        suffix = f"SVDQuant-W4A4-rank{rank}" if rank else (
            f"{args.format.upper()}-convrot" if args.format != "fp8" else "FP8")
        out = os.path.join(os.path.dirname(args.src), f"{stem}-{suffix}.safetensors")
    convert(args.src, out, fmt, args.groupsize, args.device, rank, args.refine_iters,
            variant=args.variant)

    if args.variant == "base":
        print("\nBase (non-distilled) model: start from ~50 steps, cfg 3.5, euler/simple, "
              "and a real negative prompt. See workflows/krea2_base_svdquant_w4a4_t2i.json.")
    elif args.variant == "turbo":
        print("\nTurbo (distilled) model: 8 steps, cfg 1.0, euler/simple. "
              "See workflows/krea2_svdquant_w4a4_t2i.json.")


if __name__ == "__main__":
    main()

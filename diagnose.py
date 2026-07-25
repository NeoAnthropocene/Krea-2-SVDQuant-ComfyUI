"""Run the SVDQuant diagnostics from a terminal, without starting ComfyUI.

Same reports as the "Krea2 SVDQuant Diagnostics" node, for people who would rather
paste a terminal block into an issue than wire up a graph.

    python diagnose.py <checkpoint.safetensors>
    python diagnose.py <checkpoint.safetensors> --mode bench --tokens 4096
    python diagnose.py --mode dispatch --no-load     # backend status only, no checkpoint

``--no-load`` answers the single most common question ("is the CUDA kernel even
active?") without reading 8 GB off disk.
"""

from __future__ import annotations

import argparse
import os
import sys


def _find_comfyui_root() -> str:
    """Locate the ComfyUI root so `comfy.*` imports work regardless of cwd.

    Same resolution order as quantize_krea2.py: $COMFYUI_PATH, then two levels up from
    this file (custom_nodes/<pkg>/), then cwd.
    """
    env = os.environ.get("COMFYUI_PATH")
    if env and os.path.isdir(os.path.join(env, "comfy")):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(here, "..", ".."))
    if os.path.isdir(os.path.join(candidate, "comfy")):
        return candidate
    return "."


_ROOT = _find_comfyui_root()
sys.path.insert(0, _ROOT)
# The package imports itself relatively (`from .svdquant_w4a4 import ...`), so it has to
# be importable as a package rather than as a pile of loose modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PKG = os.path.basename(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", nargs="?",
                        help="path to a --format svdq checkpoint (or a name under "
                             "models/diffusion_models)")
    parser.add_argument("--mode", default="dispatch",
                        choices=["dispatch", "env", "bench", "profile", "all"])
    parser.add_argument("--tokens", type=int, default=None,
                        help="sequence length to probe with; 4096 = 1024x1024")
    parser.add_argument("--no-load", action="store_true",
                        help="skip the checkpoint and report backend status only")
    args = parser.parse_args()

    diag = __import__("{}.svdquant_diag".format(_PKG), fromlist=["svdquant_diag"])
    tokens = args.tokens or diag._DEFAULT_TOKENS

    if args.no_load:
        print(_backend_status(diag))
        return 0

    if not args.checkpoint:
        parser.error("a checkpoint path is required unless --no-load is given")

    path = args.checkpoint
    if not os.path.isfile(path):
        candidate = os.path.join(_ROOT, "models", "diffusion_models", path)
        if os.path.isfile(candidate):
            path = candidate
        else:
            parser.error("no such checkpoint: {}".format(args.checkpoint))

    loader = __import__("{}.svdquant_w4a4".format(_PKG), fromlist=["svdquant_w4a4"])
    print("loading {} ...".format(path), flush=True)
    patcher = loader.load_svdquant_w4a4(path)

    modes = ["env", "dispatch", "bench"] if args.mode == "all" else [args.mode]
    for mode in modes:
        print()
        print(diag.run_report(patcher, mode, tokens), flush=True)
    return 0


def _backend_status(diag) -> str:
    """The subset of the `dispatch` report that needs no model."""
    import torch

    lines = ["torch {}  (cuda build {})".format(torch.__version__, torch.version.cuda), ""]
    if diag.ck_registry is None:
        lines.append("comfy_kitchen unavailable: {}".format(diag._CK_IMPORT_ERROR))
        return "\n".join(lines)

    for name, info in sorted(diag.ck_registry.list_backends().items()):
        lines.append("{:<8} available={:<5} disabled={:<5} implements {}={:<5} reason={}".format(
            name, str(info["available"]), str(info["disabled"]), diag._FUNC,
            str(diag._FUNC in info["capabilities"]), info["unavailable_reason"] or "-"))

    active = [n for n, i in diag.ck_registry.list_backends().items()
              if i["available"] and not i["disabled"] and diag._FUNC in i["capabilities"]]
    lines.append("")
    if "cuda" in active:
        lines.append("cuda backend is live -- the int4 tensor-core kernel is available.")
    else:
        lines.append(
            "cuda backend is NOT live. convrot_w4a4 will fall back to {}, which unpacks "
            "int4 to bf16 in Python -- expect this checkpoint to be slower than fp8."
            .format(active or "nothing"))
        cuda_build = torch.version.cuda
        if cuda_build is None or int(str(cuda_build).split(".")[0]) < 13:
            lines.append(
                "ComfyUI disables comfy_kitchen's CUDA backend on torch built against "
                "CUDA < 13 (comfy/quant_ops.py). Install a cu130 or newer torch build.")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

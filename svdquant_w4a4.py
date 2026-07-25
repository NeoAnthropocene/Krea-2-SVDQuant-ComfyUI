"""Loader for SVDQuant-on-native-W4A4 checkpoints.

This is SVDQuant's actual mechanism (low-rank bf16 branch + 4-bit residual) running on
ComfyUI's own ``convrot_w4a4`` kernel instead of a hand-written Triton one. Compared to
the released W4A16 checkpoint, the activations are 4-bit too, so the matmul finally runs
on hardware that is faster than bf16 rather than the same bf16 tensor cores.

The checkpoint is self-contained - it carries the quantized blocks *and* the untouched
high-precision layers - so no separate base file is needed. Everything except the 224
block linears loads through ComfyUI's normal path; only those 224 get a low-rank branch
attached on top of the native quantized Linear.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

import comfy.sd
import comfy.utils
import folder_paths

_L1 = ".svdq_l1"
_L2 = ".svdq_l2"


def attach_branch(module: torch.nn.Module, l1: torch.Tensor, l2: torch.Tensor,
                  scale: float = 1.0, kind: str = "svdq") -> None:
    """Add ``+ (x @ l2.T) @ l1.T * scale`` to a module's output, in place.

    The module is *not* replaced: swapping it for a wrapper would push its weight down a
    level in the state dict (``blocks.0.attn.wq.base.weight``), and every LoRA key map in
    ComfyUI expects ``blocks.0.attn.wq.weight``. Keeping the module identity keeps those
    paths - and therefore the rest of the ecosystem - intact.

    The branch tensors are non-persistent buffers: they move with ``.to(device)`` but stay
    out of ``state_dict``, so they cannot confuse key matching either.
    """
    if not hasattr(module, "_branch_specs"):
        module._branch_specs = []
        original = module.forward

        def forward(x, *args, **kwargs):
            y = original(x, *args, **kwargs)
            for idx, mult, _ in module._branch_specs:
                a1 = getattr(module, "_br_l1_{}".format(idx)).to(dtype=x.dtype)
                a2 = getattr(module, "_br_l2_{}".format(idx)).to(dtype=x.dtype)
                y = y + F.linear(F.linear(x, a2), a1) * mult
            return y

        module.forward = forward

    idx = len(module._branch_specs)
    module.register_buffer("_br_l1_{}".format(idx), l1.contiguous(), persistent=False)
    module.register_buffer("_br_l2_{}".format(idx), l2.contiguous(), persistent=False)
    module._branch_specs.append((idx, float(scale), kind))


def clear_branches(module: torch.nn.Module, kind: str) -> None:
    """Drop branches of one kind (used to make LoRA application idempotent)."""
    specs = getattr(module, "_branch_specs", None)
    if not specs:
        return
    module._branch_specs = [s for s in specs if s[2] != kind]


def _set_submodule(root: torch.nn.Module, dotted: str, module: torch.nn.Module) -> None:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


def _get_submodule(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    module = root
    for p in dotted.split("."):
        module = module[int(p)] if p.isdigit() else getattr(module, p)
    return module


def _shield_from_dynamo(module: torch.nn.Module) -> None:
    """Let torch.compile skip the quantized kernel instead of failing on it.

    Dynamo cannot trace ``F.linear(x, QuantizedTensor)`` -- the comfy_kitchen kernel is
    not registered as an opaque custom op, so fake-tensor tracing raises. Marking the
    call as a graph break lets inductor still fuse everything around it (norms,
    modulation, RoPE), which is a third of the step time.
    """
    try:
        module.forward = torch._dynamo.disable(module.forward)
    except Exception:
        pass


def load_svdquant_w4a4(path: str, model_options: dict | None = None,
                       compile_safe: bool = True):
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)

    branches: dict[str, dict[str, torch.Tensor]] = {}
    for key in list(sd.keys()):
        for suffix, slot in ((_L1, "l1"), (_L2, "l2")):
            if key.endswith(suffix):
                branches.setdefault(key[: -len(suffix)], {})[slot] = sd.pop(key)
                break
    if not branches:
        raise ValueError(
            "{} carries no svdq_l1/svdq_l2 tensors - it is a plain quantized checkpoint, "
            "load it with UNETLoader instead".format(path)
        )

    patcher = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options=model_options or {}, metadata=metadata, disable_dynamic=True
    )
    if patcher is None:
        raise RuntimeError("could not detect a model in {}".format(path))

    diffusion_model = patcher.model.diffusion_model
    attached = 0
    for layer, parts in branches.items():
        if "l1" not in parts or "l2" not in parts:
            continue
        base = _get_submodule(diffusion_model, layer)
        if compile_safe:
            _shield_from_dynamo(base)
        attach_branch(base, parts["l1"], parts["l2"], kind="svdq")
        attached += 1

    logging.info("[krea2-svdquant] w4a4 + low-rank: attached %d branches (rank %d)",
                 attached, int(next(iter(branches.values()))["l1"].shape[1]))
    return patcher


class Krea2SVDQuantW4A4Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "A checkpoint produced by quantize_krea2.py --format svdq"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "advanced/loaders"
    TITLE = "Krea2 SVDQuant W4A4 Loader"
    DESCRIPTION = ("Loads a W4A4 + low-rank (SVDQuant) Krea2 checkpoint. Self-contained: "
                   "no separate base model needed.")

    def load(self, model_name):
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        return (load_svdquant_w4a4(path),)


NODE_CLASS_MAPPINGS = {"Krea2SVDQuantW4A4Loader": Krea2SVDQuantW4A4Loader}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2SVDQuantW4A4Loader": "Krea2 SVDQuant W4A4 Loader"}

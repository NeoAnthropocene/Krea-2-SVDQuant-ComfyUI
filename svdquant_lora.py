"""LoRA support for Krea2 SVDQuant models.

A normal `LoraLoaderModelOnly` cannot patch the quantized layers: ComfyUI applies LoRA
by adding `down @ up` onto a module's `.weight`, and `SVDQuantLinear` has no `.weight` -
it has packed INT4 buffers plus a low-rank branch. Those 224 patches would silently do
nothing.

Krea2 LoRAs (ai-toolkit) target both kinds of layer:

* `diffusion_model.blocks.N.{attn,mlp}.*`  -> quantized, needs an adapter branch
* `diffusion_model.txtfusion.*`            -> ordinary Linear, ComfyUI patches it fine

So this node splits the LoRA: quantized layers get an inference-only LoRA branch
attached to a *replacement module* registered through `add_object_patch` (which
ComfyUI reverts after the run, so the cached base model is never mutated), and
everything else is handed to ComfyUI's normal patching path.

Stacking works: each node reads the currently patched module via `get_model_object`,
carries over its existing branches, and adds its own.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.utils
import folder_paths

from krea2_svdquant.runtime.linear import SVDQuantLinear

try:
    from .fast_kernel import FastSVDQuantLinear
except ImportError:  # running as a plain module
    from fast_kernel import FastSVDQuantLinear

_DOWN_SUFFIXES = (".lora_A.weight", ".lora_down.weight", ".lora.down.weight")
_UP_SUFFIXES = (".lora_B.weight", ".lora_up.weight", ".lora.up.weight")
_PREFIX = "diffusion_model."


class LoraBranch(torch.nn.Module):
    """``x -> (x @ down.T) @ up.T * strength * (alpha / rank)``.

    Kept separate from ``krea2_svdquant``'s own adapter because that one never moves
    its weights across devices, and ComfyUI moves the model after the patch is applied.
    """

    def __init__(self, down: torch.Tensor, up: torch.Tensor, strength: float,
                 alpha: float | None = None):
        super().__init__()
        rank = int(down.shape[0])
        self.register_buffer("down", down.contiguous(), persistent=False)
        self.register_buffer("up", up.contiguous(), persistent=False)
        self.multiplier = float(strength) * (float(alpha) / rank if alpha else 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down.device != x.device:
            self.down = self.down.to(x.device)
            self.up = self.up.to(x.device)
        down = self.down.to(dtype=x.dtype)
        up = self.up.to(dtype=x.dtype)
        return F.linear(F.linear(x, down), up) * self.multiplier


class LoraSVDQuantLinear(FastSVDQuantLinear):
    """SVDQuant linear with extra LoRA branches summed onto the output."""

    def __init__(self, state, backend, branches):
        super().__init__(state, backend=backend)
        self.branches = torch.nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        for branch in self.branches:
            y = y + branch(x)
        return y


def _split_lora(lora_sd, quant_layers: set[str]):
    """Return ({layer: {down, up, alpha}}, leftover_state_dict)."""
    pairs: dict[str, dict] = {}
    consumed: set[str] = set()

    def layer_of(key, suffixes):
        for suffix in suffixes:
            if key.endswith(suffix):
                name = key[: -len(suffix)]
                if name.startswith(_PREFIX):
                    name = name[len(_PREFIX):]
                return name
        return None

    for key in list(lora_sd.keys()):
        for suffixes, slot in ((_DOWN_SUFFIXES, "down"), (_UP_SUFFIXES, "up")):
            name = layer_of(key, suffixes)
            if name is not None and name in quant_layers:
                pairs.setdefault(name, {})[slot] = lora_sd[key]
                consumed.add(key)
                alpha_key = key.rsplit(".lora", 1)[0] + ".alpha"
                if alpha_key in lora_sd:
                    pairs[name]["alpha"] = float(lora_sd[alpha_key].item())
                    consumed.add(alpha_key)
                break

    leftover = {k: v for k, v in lora_sd.items() if k not in consumed}
    return pairs, leftover


def apply_svdquant_lora(patcher, lora_sd, strength: float):
    diffusion_model = patcher.model.diffusion_model

    # Two model flavours: the released W4A16 checkpoint (SVDQuantLinear modules) and the
    # W4A4 one built by quantize_krea2.py (ComfyUI quantized Linears carrying branches).
    w4a16 = {name for name, m in diffusion_model.named_modules()
             if isinstance(m, SVDQuantLinear)}
    w4a4 = {name for name, m in diffusion_model.named_modules()
            if hasattr(m, "_branch_specs")}
    quant_layers = w4a16 | w4a4

    pairs, leftover = _split_lora(lora_sd, quant_layers)

    applied = 0
    for layer, parts in pairs.items():
        if "down" not in parts or "up" not in parts:
            continue
        if layer in w4a4:
            from .svdquant_w4a4 import attach_branch
            module = diffusion_model.get_submodule(layer)
            device = next(
                (b.device for b in module.buffers()), torch.device("cpu"))
            rank = int(parts["down"].shape[0])
            alpha = parts.get("alpha")
            mult = strength * (float(alpha) / rank if alpha else 1.0)
            attach_branch(
                module,
                parts["up"].to(device=device, dtype=torch.bfloat16),
                parts["down"].to(device=device, dtype=torch.bfloat16),
                scale=mult, kind="lora",
            )
            applied += 1
            continue

        key = _PREFIX + layer
        current = patcher.get_model_object(key)
        branches = list(getattr(current, "branches", []))
        device = current.qweight.device
        branches.append(LoraBranch(
            parts["down"].to(device=device, dtype=torch.bfloat16),
            parts["up"].to(device=device, dtype=torch.bfloat16),
            strength, parts.get("alpha"),
        ))
        patcher.add_object_patch(
            key, LoraSVDQuantLinear(current.state, current.backend, branches)
        )
        applied += 1

    # Everything the quantized layers did not claim (txtfusion, etc.) goes through
    # ComfyUI's normal LoRA path.
    patched_normally = 0
    if leftover:
        key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
        loaded = comfy.lora.load_lora(leftover, key_map, log_missing=False)
        if loaded:
            patcher.add_patches(loaded, strength)
            patched_normally = len(loaded)

    return applied, patched_normally


class Krea2SVDQuantLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Output of the Krea2 SVDQuant Loader."}),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "advanced/loaders"
    TITLE = "Krea2 SVDQuant LoRA Loader"
    DESCRIPTION = ("Applies a LoRA to a Krea2 SVDQuant model. The standard LoRA loader "
                   "cannot patch the quantized layers because they have no .weight tensor.")

    def load_lora(self, model, lora_name, strength):
        if strength == 0:
            return (model,)
        patcher = model.clone()

        # The W4A4 branches live on the shared model, so re-apply the whole stack from
        # scratch each time instead of appending blindly. That keeps this node idempotent
        # when a strength changes, and still stacks correctly when nodes are chained.
        stack = list(getattr(model, "krea2_lora_stack", [])) + [(lora_name, strength)]
        patcher.krea2_lora_stack = stack

        from .svdquant_w4a4 import clear_branches
        for module in patcher.model.diffusion_model.modules():
            clear_branches(module, "lora")

        quantized = normal = 0
        for name, amount in stack:
            path = folder_paths.get_full_path_or_raise("loras", name)
            lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
            q, n = apply_svdquant_lora(patcher, lora_sd, amount)
            quantized += q
            normal += n
        if quantized == 0 and normal == 0:
            raise ValueError(
                "no layer of {} matched this model; is it a Krea2 LoRA?".format(lora_name)
            )
        logging.info("[krea2-svdquant] LoRA stack %s -> %d quantized layers, %d normal layers",
                     [f"{n}@{a:.2f}" for n, a in stack], quantized, normal)
        return (patcher,)


NODE_CLASS_MAPPINGS = {"Krea2SVDQuantLoraLoader": Krea2SVDQuantLoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2SVDQuantLoraLoader": "Krea2 SVDQuant LoRA Loader"}

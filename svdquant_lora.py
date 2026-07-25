"""LoRA support for Krea2 W4A4 quantized models (from ``quantize_krea2.py``).

A normal `LoraLoaderModelOnly` cannot patch these quantized layers correctly: ComfyUI
applies a LoRA by adding `down @ up` onto a module's `.weight`, but here `.weight` is a
`QuantizedTensor` -- patching it that way would mean dequantize -> add -> requantize,
losing the format, or in practice just silently missing the layer.

Krea2 LoRAs (ai-toolkit) target both kinds of layer:

* `diffusion_model.blocks.N.{attn,mlp}.*`  -> quantized, needs a parallel branch
* `diffusion_model.txtfusion.*`            -> ordinary Linear, ComfyUI patches it fine

So this node splits the LoRA: quantized layers get the LoRA attached as an extra
parallel branch (mathematically identical for a linear layer: `(W + BA)x == Wx + B(Ax)`,
so the quantized weight itself is never touched), and everything else is handed to
ComfyUI's normal patching path.

Stacking works: chaining multiple of these nodes re-applies the whole LoRA stack from
scratch each time, so strengths can change without leftover state from a previous value.
"""

from __future__ import annotations

import logging

import torch

import comfy.lora
import comfy.utils
import folder_paths

_DOWN_SUFFIXES = (".lora_A.weight", ".lora_down.weight", ".lora.down.weight")
_UP_SUFFIXES = (".lora_B.weight", ".lora_up.weight", ".lora.up.weight")
_PREFIX = "diffusion_model."


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
    from .svdquant_w4a4 import attach_branch

    diffusion_model = patcher.model.diffusion_model
    quant_layers = {name for name, m in diffusion_model.named_modules()
                    if hasattr(m, "_branch_specs")}

    pairs, leftover = _split_lora(lora_sd, quant_layers)

    applied = 0
    for layer, parts in pairs.items():
        if "down" not in parts or "up" not in parts:
            continue
        module = diffusion_model.get_submodule(layer)
        device = next((b.device for b in module.buffers()), None)
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
                "model": ("MODEL", {"tooltip": "Output of the Krea2 SVDQuant W4A4 Loader."}),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "advanced/loaders"
    TITLE = "Krea2 SVDQuant LoRA Loader"
    DESCRIPTION = ("Applies a LoRA to a Krea2 W4A4 quantized model. The standard LoRA "
                   "loader silently skips the quantized layers on these models.")

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

"""LoRA support for Krea2 W4A4 quantized models (from ``quantize_krea2.py``).

A normal `LoraLoaderModelOnly` cannot patch these quantized layers correctly: ComfyUI
applies a LoRA by adding `down @ up` onto a module's `.weight`, but here `.weight` is a
`QuantizedTensor` -- patching it that way means dequantize -> add -> requantize, which
throws the 4-bit format away and re-quantizes the LoRA delta along with it.

Krea2 LoRAs (ai-toolkit) target both kinds of layer:

* `diffusion_model.blocks.N.{attn,mlp}.*`  -> quantized, needs a parallel branch
* `diffusion_model.txtfusion.*`            -> ordinary Linear, ComfyUI patches it fine

So this node splits the LoRA: quantized layers get the LoRA attached as an extra
parallel branch (mathematically identical for a linear layer: `(W + BA)x == Wx + B(Ax)`,
so the quantized weight itself is never touched), and everything else is handed to
ComfyUI's normal patching path.

The branch is installed through ``ModelPatcher.add_object_patch`` rather than by mutating
the module. That is what makes the node behave like every other ComfyUI node: `clone()`
shares the underlying `nn.Module`, so writing the LoRA onto the module directly would
leak it into every other branch of the graph that came off the same loader, and would
survive past the sampling run. Object patches are applied in `patch_model` and reverted
afterwards, per patcher.

Stacking works: chaining multiple of these nodes re-applies the whole LoRA stack from
scratch each time, so strengths can change without leftover state from a previous value.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

import torch

import comfy.lora
import comfy.utils
import folder_paths

from .svdquant_diag import _CATEGORY, quantized_linears
from .svdquant_w4a4 import add_low_rank, has_branch

_DOWN_SUFFIXES = (".lora_A.weight", ".lora_down.weight", ".lora.down.weight", ".lora_A", ".lora_down")
_UP_SUFFIXES = (".lora_B.weight", ".lora_up.weight", ".lora.up.weight", ".lora_B", ".lora_up")
_LOKR_W1_SUFFIXES = (".lokr_w1", ".lokr_w1.weight", ".lokr_w1_a", ".lokr_w1_a.weight", ".hada_w1", ".hada_w1.weight", ".lokr.down.weight")
_LOKR_W2_SUFFIXES = (".lokr_w2", ".lokr_w2.weight", ".lokr_w2_a", ".lokr_w2_a.weight", ".hada_w2", ".hada_w2.weight", ".lokr.up.weight")
# ComfyUI's native prefix, plus the diffusers one some trainers write in front of the very
# same native module names (`transformer.blocks.N.attn.wq`). That hybrid matches nothing in
# `comfy/lora.py` -- its `transformer.` entries are keyed on *diffusers* module names
# (`transformer.transformer_blocks.N.attn.to_q`) -- so the stock loader logs "lora key not
# loaded" for every key and applies nothing at all. Accepting it here is the difference
# between the LoRA working and it silently doing nothing.
_PREFIXES = ("diffusion_model.", "transformer.")

_DIFFUSERS_NAME_MAP = [
    ("transformer_blocks.", "blocks."),
    (".attn.to_q", ".attn.wq"),
    (".attn.to_k", ".attn.wk"),
    (".attn.to_v", ".attn.wv"),
    (".attn.to_out.0", ".attn.wo"),
    (".attn.to_gate", ".attn.gate"),
    (".ff.gate", ".mlp.gate"),
    (".ff.up", ".mlp.up"),
    (".ff.down", ".mlp.down"),
]


def _normalize_layer_name(name: str, quant_layers: set[str]) -> str:
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    if name in quant_layers:
        return name

    mapped = name
    for old, new in _DIFFUSERS_NAME_MAP:
        if old.startswith(".") or old.endswith("."):
            if old in mapped:
                mapped = mapped.replace(old, new)
        else:
            if mapped.startswith(old):
                mapped = new + mapped[len(old):]

    return mapped


# Reloading a 300 MB LoRA off disk on every graph execution is pure latency when the
# usual edit is a strength slider. Keyed on mtime+size so editing the file invalidates.
_LORA_CACHE: OrderedDict[str, tuple[tuple, dict]] = OrderedDict()
_LORA_CACHE_MAX = 4


def _load_lora_cached(path: str) -> dict:
    stat = os.stat(path)
    stamp = (stat.st_mtime_ns, stat.st_size)
    hit = _LORA_CACHE.get(path)
    if hit is not None and hit[0] == stamp:
        # Without this the eviction below is insertion-ordered (FIFO), so cycling five LoRAs
        # can drop the one being used every run.
        _LORA_CACHE.move_to_end(path)
        return hit[1]
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    if len(_LORA_CACHE) >= _LORA_CACHE_MAX:
        _LORA_CACHE.pop(next(iter(_LORA_CACHE)))
    _LORA_CACHE[path] = (stamp, sd)
    return sd


def _split_lora(lora_sd, quant_layers: set[str]):
    """Return ({layer: {down, up, alpha}}, leftover_state_dict)."""
    pairs: dict[str, dict] = {}
    consumed: set[str] = set()

    def layer_of(key, suffixes):
        for suffix in suffixes:
            if key.endswith(suffix):
                raw_name = key[: -len(suffix)]
                return _normalize_layer_name(raw_name, quant_layers)
        return None

    for key in list(lora_sd.keys()):
        for suffixes, slot in (
            (_DOWN_SUFFIXES, "down"),
            (_UP_SUFFIXES, "up"),
            (_LOKR_W1_SUFFIXES, "lokr_w1"),
            (_LOKR_W2_SUFFIXES, "lokr_w2"),
        ):
            name = layer_of(key, suffixes)
            if name is not None and name in quant_layers:
                pairs.setdefault(name, {})[slot] = lora_sd[key]
                consumed.add(key)
                prefix_key = key.rsplit(".", 1)[0]
                alpha_key = prefix_key + ".alpha"
                if alpha_key in lora_sd and alpha_key not in consumed:
                    pairs[name]["alpha"] = float(lora_sd[alpha_key].item())
                    consumed.add(alpha_key)
                dim_key = prefix_key + ".dim"
                if dim_key in lora_sd and dim_key not in consumed:
                    pairs[name]["dim"] = float(lora_sd[dim_key].item())
                    consumed.add(dim_key)
                break

    leftover = {k: v for k, v in lora_sd.items() if k not in consumed}
    return pairs, leftover


def _make_lora_forward(module, standard_factors: list[tuple[torch.Tensor, torch.Tensor]],
                       lokr_factors: list[tuple[torch.Tensor, torch.Tensor, float]]):
    """A forward that is "quantized weight + svdq branch (if any) + LoRAs".

    Built on `module._krea2_forward` rather than `module.forward` so that it composes with
    the svdq branch but never with a previously installed LoRA patch -- each patcher owns
    the whole LoRA stack and rebuilds it from scratch.
    """
    base = getattr(module, "_krea2_forward", None) or module.forward
    has_std = len(standard_factors) > 0
    if has_std:
        std_l1, std_l2 = _stack_factors(standard_factors)

    def forward(x, *args, **kwargs):
        y = base(x, *args, **kwargs)
        if has_std:
            y = add_low_rank(y, x, std_l1, std_l2)
        for w1, w2, mult in lokr_factors:
            a1 = comfy.model_management.cast_to(w1, x.dtype, x.device)
            a2 = comfy.model_management.cast_to(w2, x.dtype, x.device)
            orig_shape = x.shape
            n1, n2 = a1.shape[1], a2.shape[1]
            m1, m2 = a1.shape[0], a2.shape[0]
            x_reshaped = x.view(-1, n1, n2)
            lokr_out = torch.matmul(torch.matmul(a1, x_reshaped), a2.T).view(orig_shape[:-1] + (m1 * m2,))
            y = y + lokr_out * mult
        return y

    return forward


def _stack_factors(factors: list[tuple[torch.Tensor, torch.Tensor]]):
    """Collapse a LoRA stack for one layer into a single (l1, l2) pair.

    With the strength already folded into each l2, ``sum_i l1_i @ l2_i`` is exactly
    ``cat(l1, dim=1) @ cat(l2, dim=0)``, so an N-LoRA stack costs one pair of GEMMs per
    step instead of N.
    """
    if len(factors) == 1:
        return factors[0]
    l1 = torch.cat([f[0] for f in factors], dim=1)
    l2 = torch.cat([f[1] for f in factors], dim=0)
    return l1.contiguous(), l2.contiguous()


def collect_svdquant_lora(patcher, lora_sd, strength: float, quant_layers: set[str],
                          device, dtype, into_std: dict, into_lokr: dict, patch_leftover: bool = True):
    """Accumulate this LoRA's quantized-layer factors into `into_std` and `into_lokr`; patch the rest normally.

    `patch_leftover=False` collects the quantized side only and leaves the non-quantized
    layers alone -- see `load_lora` for why the two sides of the stack are handled
    differently.

    Returns the number of layers matched on each path and whether any leftover patch
    targeted a quantized weight.
    """
    pairs, leftover = _split_lora(lora_sd, quant_layers)

    applied = 0
    for layer, parts in pairs.items():
        if "down" in parts and "up" in parts:
            rank = int(parts["down"].shape[0])
            alpha = parts.get("alpha")
            if alpha is None or float(alpha) > 10000.0 or float(alpha) <= 0:
                mult = strength
            else:
                mult = strength * (float(alpha) / rank)
            l1 = parts["up"].to(device=device, dtype=dtype)
            l2 = parts["down"].to(device=device, dtype=dtype) * mult
            into_std.setdefault(layer, []).append((l1.contiguous(), l2.contiguous()))
            applied += 1
        elif "lokr_w1" in parts and "lokr_w2" in parts:
            w1 = parts["lokr_w1"].to(device=device, dtype=dtype)
            w2 = parts["lokr_w2"].to(device=device, dtype=dtype)
            alpha = parts.get("alpha")
            dim = parts.get("dim")
            rank = int(dim) if dim is not None else (int(w2.shape[0]) if w2.ndim == 2 else 1)
            if alpha is None or float(alpha) > 10000.0 or float(alpha) <= 0:
                mult = strength
            else:
                mult = strength * (float(alpha) / rank)
            into_lokr.setdefault(layer, []).append((w1, w2, mult))
            applied += 1

    if not patch_leftover:
        return applied, 0, False

    # Everything the quantized layers did not claim (txtfusion, etc.) goes through
    # ComfyUI's normal LoRA path.
    patched_normally = 0
    patched_quantized_normally = False
    if leftover:
        key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
        loaded = comfy.lora.load_lora(leftover, key_map, log_missing=False)
        if loaded:
            patcher.add_patches(loaded, strength)
            patched_normally = len(loaded)
            for patch_key in loaded.keys():
                key_str = str(patch_key)
                for ql in quant_layers:
                    if ql in key_str:
                        patched_quantized_normally = True
                        break

    return applied, patched_normally, patched_quantized_normally


class Krea2SVDQuantLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Output of the Krea2 SVDQuant W4A4 Loader, or any Krea2 "
                               "checkpoint with convrot_w4a4 quantized blocks.",
                }),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "A Krea2 LoRA. Targets 'blocks.N.{attn,mlp}.*' under either "
                               "prefix ('diffusion_model.' or 'transformer.') for the "
                               "quantized blocks; anything else it carries (txtfusion "
                               "etc.) goes through ComfyUI's normal path.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "0 passes the model through untouched. Negative values invert "
                               "the LoRA. Chain more of these nodes to stack LoRAs - the "
                               "whole stack is rebuilt each time, so strengths stay exact.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("The model with the LoRA attached as a parallel branch.",)
    FUNCTION = "load_lora"
    CATEGORY = _CATEGORY
    TITLE = "Krea2 SVDQuant LoRA Loader"
    DESCRIPTION = ("Applies a LoRA to a Krea2 W4A4 quantized model as a parallel low-rank "
                   "branch, leaving the 4-bit weight untouched. The stock loader has to "
                   "dequantize, add and requantize, which puts the LoRA delta through 4-bit "
                   "quantization along with the weight.")

    def load_lora(self, model, lora_name, strength):
        if strength == 0:
            return (model,)
        patcher = model.clone()

        # Re-apply the whole stack from scratch rather than appending to whatever the
        # upstream node left behind. That keeps this node idempotent when a strength
        # changes, and still stacks correctly when nodes are chained.
        stack = list(getattr(model, "krea2_lora_stack", [])) + [(lora_name, strength)]
        patcher.krea2_lora_stack = stack

        diffusion_model = patcher.model.diffusion_model
        # Every convrot_w4a4 layer needs the parallel-branch treatment, whether or not it
        # already carries an svdq branch. Keying this off `has_branch` instead would come up
        # empty on a no-low-rank checkpoint, silently route the whole LoRA to ComfyUI's
        # normal path, and there it cannot patch a QuantizedTensor weight at all -- the exact
        # failure this node exists to prevent, with no error to show for it.
        quant_modules = dict(quantized_linears(diffusion_model))
        quant_layers = set(quant_modules)
        unbranched = sum(1 for m in quant_modules.values() if not has_branch(m))

        # Attach on the offload device and let the forward stage them: matching whatever
        # device the module happens to be on right now would bake in a placement that
        # ComfyUI is free to change before the first step.
        device = patcher.offload_device
        dtype = patcher.model.get_dtype()

        collected_std: dict[str, list] = {}
        collected_lokr: dict[str, list] = {}
        quantized = normal = 0
        patched_quantized_normally = False
        newest = len(stack) - 1
        for i, (name, amount) in enumerate(stack):
            path = folder_paths.get_full_path_or_raise("loras", name)
            q, n, pqn = collect_svdquant_lora(
                patcher, _load_lora_cached(path), amount, quant_layers, device, dtype,
                collected_std, collected_lokr, patch_leftover=(i == newest))
            if i == newest:
                quantized, normal, patched_quantized_normally = q, n, pqn

        # Both guards are about the LoRA this node just added, not the stack total: an
        # upstream LoRA matching plenty of layers must not excuse this one matching none.
        if quantized == 0 and normal == 0:
            raise ValueError(
                "no layer of {} matched this model ({} quantized layers were available); "
                "is it a Krea2 LoRA?".format(lora_name, len(quant_layers))
            )
        if quantized == 0 and patched_quantized_normally:
            raise ValueError(
                "{} matched {} non-quantized layers but none of the {} quantized ones. "
                "ComfyUI's normal LoRA path cannot patch a quantized weight, so the blocks "
                "would silently go unchanged. Check that the LoRA targets "
                "'[diffusion_model.|transformer.]blocks.N.{{attn,mlp}}.*'.".format(
                    lora_name, normal, len(quant_layers))
            )

        all_layers = set(collected_std.keys()) | set(collected_lokr.keys())
        for layer in all_layers:
            module = diffusion_model.get_submodule(layer)
            std_factors = collected_std.get(layer, [])
            lokr_factors = collected_lokr.get(layer, [])
            patcher.add_object_patch(
                "diffusion_model.{}.forward".format(layer),
                _make_lora_forward(module, std_factors, lokr_factors),
            )

        logging.info("[krea2-svdquant] LoRA stack %s -> %d quantized layers branched; "
                     "%s added %d quantized, %d normal layers%s",
                     [f"{n}@{a:.2f}" for n, a in stack], len(all_layers), lora_name,
                     quantized, normal,
                     " (no-low-rank checkpoint: {} quantized layers carry no svdq branch)"
                     .format(unbranched) if unbranched else "")
        return (patcher,)


NODE_CLASS_MAPPINGS = {"Krea2SVDQuantLoraLoader": Krea2SVDQuantLoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2SVDQuantLoraLoader": "Krea2 SVDQuant LoRA Loader"}

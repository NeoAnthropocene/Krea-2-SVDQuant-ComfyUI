"""Load a Krea2 SVDQuant (W4A16 + BF16 low-rank) transformer checkpoint into ComfyUI.

The published checkpoint is *transformer-linear only*:
it holds 224 quantized ``nn.Linear`` layers (28 blocks x 8 layers) and nothing else -
no embeddings, norms, modulation, text fusion stack or final layer. Those still have to
come from a normal Krea2 Turbo checkpoint.

So the loader:

1. reads the base checkpoint but *skips* every tensor belonging to a layer that the
   SVDQuant file replaces (that is ~12 GB of the 12 GB file, so the base costs almost
   nothing to load),
2. builds the ComfyUI ``SingleStreamDiT`` on the ``meta`` device so those 12B params are
   never allocated,
3. swaps the 224 linears for ``SVDQuantLinear`` modules built straight from the
   quantized file,
4. materializes the remaining (small) weights with ``assign=True``.

Checkpoint layer names are diffusers-style; ComfyUI's krea2 implementation uses its own
names. The mapping is exact, see ``_SUFFIX_MAP``.
"""

from __future__ import annotations

import json
import logging
import os
import re

import torch
from safetensors import safe_open

import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths

from krea2_svdquant.config import BackendKind
from krea2_svdquant.quant.svd import SVDQuantLinearState
from krea2_svdquant.runtime.linear import SVDQuantLinear
from krea2_svdquant.runtime.replace import replace_module

try:
    from .fast_kernel import FastSVDQuantLinear
except ImportError:  # running as a plain module (tests/benchmarks)
    from fast_kernel import FastSVDQuantLinear

# "tensorcore" is our own bf16 tensor-core kernel; the rest map onto BackendKind.
TENSORCORE = "tensorcore"
BACKENDS = [TENSORCORE, "auto", "triton_generic", "pytorch_sim",
            "triton_blackwell", "gluon_blackwell"]

# diffusers name (in svdquant_config.json) -> ComfyUI comfy/ldm/krea2/model.py name
_SUFFIX_MAP = {
    "attn.to_q": "attn.wq",
    "attn.to_k": "attn.wk",
    "attn.to_v": "attn.wv",
    "attn.to_gate": "attn.gate",
    "attn.to_out.0": "attn.wo",
    "ff.gate": "mlp.gate",
    "ff.up": "mlp.up",
    "ff.down": "mlp.down",
}

# Tensor-name suffixes that belong to a layer rather than being part of its name.
_PARAM_SUFFIXES = (
    ".weight",
    ".bias",
    ".scale_weight",
    ".scale_input",
    ".weight_scale",
    ".input_scale",
    ".comfy_quant",
)

_REQUIRED_QUANT_TENSORS = ("qweight_packed", "weight_scales", "smooth_scale", "l1", "l2")


def comfy_layer_name(diffusers_name: str) -> str | None:
    """``transformer_blocks.3.attn.to_q`` -> ``blocks.3.attn.wq``."""
    match = re.match(r"^transformer_blocks\.(\d+)\.(.+)$", diffusers_name)
    if match is None:
        return None
    mapped = _SUFFIX_MAP.get(match.group(2))
    if mapped is None:
        return None
    return "blocks.{}.{}".format(match.group(1), mapped)


def _layer_of(tensor_name: str) -> str:
    for suffix in _PARAM_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return tensor_name


def find_svdquant_config(model_path: str, override: str = "") -> str:
    if override:
        if not os.path.isabs(override):
            override = os.path.join(os.path.dirname(model_path), override)
        if not os.path.exists(override):
            raise FileNotFoundError("svdquant config not found: {}".format(override))
        return override
    candidate = os.path.join(os.path.dirname(model_path), "svdquant_config.json")
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            "svdquant_config.json not found next to {}. It ships with the checkpoint "
            "and is required to know each layer's rank/group size.".format(model_path)
        )
    return candidate


def _load_base_without(path: str, skip_layers: set[str]):
    """Read the base checkpoint, dropping every tensor of a replaced layer."""
    state_dict = {}
    skipped = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            if _layer_of(key) in skip_layers:
                skipped += 1
                continue
            state_dict[key] = handle.get_tensor(key)
    return state_dict, metadata, skipped


def _prune_quant_metadata(metadata, skip_layers: set[str]):
    """Drop replaced layers from ``_quantization_metadata`` so ComfyUI does not
    build mixed-precision ops (or expect scale tensors) for modules we replace."""
    if not metadata or "_quantization_metadata" not in metadata:
        return metadata
    parsed = json.loads(metadata["_quantization_metadata"])
    layers = parsed.get("layers", {})
    parsed["layers"] = {k: v for k, v in layers.items() if k not in skip_layers}
    out = dict(metadata)
    out["_quantization_metadata"] = json.dumps(parsed)
    return out


def _retarget_factory_devices(root: torch.nn.Module, device) -> None:
    for module in root.modules():
        factory_kwargs = getattr(module, "factory_kwargs", None)
        if isinstance(factory_kwargs, dict):
            current = factory_kwargs.get("device", None)
            if current is None or torch.device(current).type == "meta":
                factory_kwargs["device"] = device


def _resolve_backend(backend: str):
    """Return (module_class, BackendKind used for the fallback path)."""
    if backend == TENSORCORE:
        return FastSVDQuantLinear, BackendKind.AUTO
    return SVDQuantLinear, BackendKind(backend)


def _build_svdquant_module(handle, diffusers_name: str, layer_meta: dict,
                           default_group: int, backend, module_cls=SVDQuantLinear) -> SVDQuantLinear:
    key = diffusers_name.replace(".", "__")
    tensors = {}
    for suffix in _REQUIRED_QUANT_TENSORS:
        name = "{}.{}".format(key, suffix)
        try:
            tensors[suffix] = handle.get_tensor(name)
        except Exception as exc:
            raise KeyError("missing tensor {} for layer {}".format(name, diffusers_name)) from exc

    bias_name = "{}.bias".format(key)
    try:
        bias = handle.get_tensor(bias_name)
    except Exception:
        bias = None

    out_features, in_features = (int(v) for v in layer_meta["shape"])
    qweight = tensors["qweight_packed"]
    state = SVDQuantLinearState(
        smooth_scale=tensors["smooth_scale"],
        qweight=qweight,
        weight_scales=tensors["weight_scales"],
        l1=tensors["l1"],
        l2=tensors["l2"],
        bias=bias,
        group_size=int(layer_meta.get("group_size", default_group)),
        original_shape=(out_features, in_features),
        qweight_packed=True,
        padded_in_features=int(qweight.shape[1]) * 2,
    )
    return module_cls(state, backend=backend)


def load_krea2_svdquant(svdquant_path: str, base_path: str, config_path: str,
                        backend: str = TENSORCORE, out_chunk: int = 0,
                        model_options: dict | None = None) -> comfy.model_patcher.ModelPatcher:
    model_options = model_options or {}
    module_cls, backend_kind = _resolve_backend(backend)

    with open(config_path, "r", encoding="utf-8") as fh:
        quant_config = json.load(fh)

    layers = quant_config.get("layers", {})
    if not layers:
        raise ValueError("{} lists no layers".format(config_path))

    default_group = int(quant_config.get("group_size", 128))

    # Map every quantized layer onto its ComfyUI module path.
    layer_map = {}
    for diffusers_name, layer_meta in layers.items():
        target = comfy_layer_name(diffusers_name)
        if target is None:
            raise ValueError(
                "cannot map checkpoint layer {!r} onto ComfyUI's krea2 model".format(diffusers_name)
            )
        layer_map[diffusers_name] = (target, layer_meta)

    skip_layers = {target for target, _ in layer_map.values()}

    # --- base checkpoint, minus everything we are about to replace -------------
    base_sd, metadata, skipped = _load_base_without(base_path, skip_layers)
    logging.info(
        "[krea2-svdquant] base %s: kept %d tensors, skipped %d replaced-layer tensors",
        os.path.basename(base_path), len(base_sd), skipped,
    )
    metadata = _prune_quant_metadata(metadata, skip_layers)

    base_sd, metadata = comfy.utils.convert_old_quants(base_sd, "", metadata=metadata)

    # Model detection reads the shape of blocks.0.attn.wq/wk to derive heads/kvheads,
    # but we deliberately did not load those. Meta tensors carry shape at zero cost.
    detect_sd = dict(base_sd)
    for diffusers_name in ("transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k"):
        target, layer_meta = layer_map[diffusers_name]
        detect_sd["{}.weight".format(target)] = torch.empty(
            tuple(int(v) for v in layer_meta["shape"]), device="meta", dtype=torch.bfloat16
        )

    model_config = comfy.model_detection.model_config_from_unet(detect_sd, "", metadata=metadata)
    if model_config is None:
        raise RuntimeError("could not detect a Krea2 model in {}".format(base_path))
    if model_config.unet_config.get("image_model") != "krea2":
        raise RuntimeError(
            "base model is {!r}, expected a Krea2 checkpoint".format(
                model_config.unet_config.get("image_model")
            )
        )
    del detect_sd

    load_device = model_options.get("load_device", comfy.model_management.get_torch_device())
    offload_device = model_options.get("offload_device", comfy.model_management.unet_offload_device())

    weight_dtype = comfy.utils.weight_dtype(base_sd)
    if model_config.quant_config is not None:
        weight_dtype = None

    unet_dtype = model_options.get("dtype", None)
    if unet_dtype is None:
        # The quantized bulk is not in base_sd, so size the dtype decision by the real
        # parameter count of the model rather than by what we happened to load.
        parameters = int(quant_config.get("num_blocks", 28)) * 434_110_464
        unet_dtype = comfy.model_management.unet_dtype(
            model_params=parameters,
            supported_dtypes=list(model_config.supported_inference_dtypes),
            weight_dtype=weight_dtype,
        )

    if model_config.quant_config is not None:
        manual_cast_dtype = comfy.model_management.unet_manual_cast(
            None, load_device, model_config.supported_inference_dtypes
        )
    else:
        manual_cast_dtype = comfy.model_management.unet_manual_cast(
            unet_dtype, load_device, model_config.supported_inference_dtypes
        )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)

    # --- build on meta so the 12B replaced params are never allocated ----------
    model = model_config.get_model(base_sd, "", device=torch.device("meta"))
    diffusion_model = model.diffusion_model

    # --- swap in the quantized linears ----------------------------------------
    replaced = 0
    with safe_open(svdquant_path, framework="pt", device="cpu") as handle:
        for diffusers_name, (target, layer_meta) in layer_map.items():
            module = _build_svdquant_module(
                handle, diffusers_name, layer_meta, default_group, backend_kind, module_cls
            )
            replace_module(diffusion_model, target, module)
            replaced += 1

    # ComfyUI's mixed-precision Linear builds its weight inside _load_from_state_dict and
    # places it on factory_kwargs["device"] -- which is still "meta" from construction.
    # Point it at the real device before loading, or those weights stay meta.
    _retarget_factory_devices(diffusion_model, offload_device)

    # --- materialize the remaining weights ------------------------------------
    to_load = model_config.process_unet_state_dict(base_sd)
    missing, unexpected = diffusion_model.load_state_dict(to_load, strict=False, assign=True)
    # Buffers of the modules we swapped in are already populated; they only show up as
    # "missing" because they are absent from the base state dict.
    missing = [k for k in missing if k.rsplit(".", 1)[0] not in skip_layers]
    if missing:
        logging.warning("[krea2-svdquant] missing keys: %s", missing)
    if unexpected:
        logging.warning("[krea2-svdquant] unexpected keys: %s", unexpected)
    del to_load, base_sd

    still_meta = [n for n, p in diffusion_model.named_parameters() if p.is_meta]
    still_meta += [n for n, b in diffusion_model.named_buffers() if b.is_meta]
    if still_meta:
        raise RuntimeError(
            "{} parameters had no weights in the base checkpoint (first few: {}). "
            "The base model does not match this SVDQuant checkpoint.".format(
                len(still_meta), still_meta[:8]
            )
        )

    if out_chunk > 0:
        os.environ["KREA2_SVDQ_OUT_CHUNK"] = str(int(out_chunk))
    else:
        os.environ.pop("KREA2_SVDQ_OUT_CHUNK", None)

    patcher = comfy.model_patcher.ModelPatcher(
        model, load_device=load_device, offload_device=offload_device
    )
    if not comfy.model_management.is_device_cpu(offload_device):
        model.to(offload_device)

    logging.info(
        "[krea2-svdquant] replaced %d linears, backend=%s, unet_dtype=%s, manual_cast=%s, size=%.2f GB",
        replaced, backend, unet_dtype, manual_cast_dtype,
        comfy.model_management.module_size(model) / (1024 ** 3),
    )
    return patcher


def _diffusion_model_list():
    return folder_paths.get_filename_list("diffusion_models")


class Krea2SVDQuantLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = _diffusion_model_list()
        return {
            "required": {
                "svdquant_model": (models, {
                    "tooltip": "The SVDQuant transformer checkpoint, e.g. transformer_svdquant.safetensors."
                }),
                "base_model": (models, {
                    "tooltip": "A normal Krea2 Turbo checkpoint. Supplies embeddings, norms, "
                               "modulation, text fusion and the final layer, which the SVDQuant "
                               "file does not contain."
                }),
                "backend": (BACKENDS, {
                    "default": TENSORCORE,
                    "tooltip": "tensorcore is a bf16 tensor-core W4A16 kernel (~3.5x faster than "
                               "the reference Triton kernel on Ampere, bit-comparable results). "
                               "The other options select krea2_svdquant's own backends."
                }),
            },
            "optional": {
                "out_chunk": ("INT", {
                    "default": 0, "min": 0, "max": 16384, "step": 512,
                    "tooltip": "Only used by the pytorch_sim fallback: dequantize this many output "
                               "channels at a time to cap transient VRAM. 0 disables chunking."
                }),
                "config_override": ("STRING", {
                    "default": "",
                    "tooltip": "Path to svdquant_config.json if it is not next to the model."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "advanced/loaders"
    TITLE = "Krea2 SVDQuant Loader"
    DESCRIPTION = ("Loads a Krea2 SVDQuant W4A16 transformer, merging it with a full Krea2 Turbo "
                   "base checkpoint for the layers the quantized file omits.")

    def load(self, svdquant_model, base_model, backend, out_chunk=0, config_override=""):
        svdquant_path = folder_paths.get_full_path_or_raise("diffusion_models", svdquant_model)
        base_path = folder_paths.get_full_path_or_raise("diffusion_models", base_model)
        if os.path.normcase(svdquant_path) == os.path.normcase(base_path):
            raise ValueError("svdquant_model and base_model must be different files")
        config_path = find_svdquant_config(svdquant_path, config_override)
        patcher = load_krea2_svdquant(
            svdquant_path, base_path, config_path, backend=backend, out_chunk=out_chunk
        )
        return (patcher,)


NODE_CLASS_MAPPINGS = {"Krea2SVDQuantLoader": Krea2SVDQuantLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2SVDQuantLoader": "Krea2 SVDQuant Loader"}

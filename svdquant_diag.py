"""Diagnostics for Krea2 SVDQuant / W4A4 checkpoints.

Answers the questions a performance or OOM report needs answered before anything
else can be said:

* which comfy_kitchen backend ``convrot_w4a4_linear`` actually dispatches to on this
  machine -- ``cuda`` is the int4 tensor-core kernel, ``eager`` is a pure-PyTorch
  int4-unpack-then-bf16-matmul that is *slower* than plain bf16;
* how ComfyUI's memory accounting sees the model, and where the weights currently live.

The first one matters more than it looks: ComfyUI disables comfy_kitchen's CUDA backend
outright when torch was built against CUDA < 13 (``comfy/quant_ops.py``), which silently
turns every quantized checkpoint into the eager path. That one fact explains most
"quantized is slower than fp8" reports and is invisible without this node.

Everything here is read-only apart from ``bench``, which needs the weights resident and
will ask ComfyUI to load the model the same way sampling would.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time

import torch
import torch.nn.functional as F

import comfy.model_management as mm
# Importing this is what applies ComfyUI's own backend gating -- notably the
# `ck.registry.disable("cuda")` it performs when torch was built against CUDA < 13.
# Without it we would query a registry ComfyUI has not finished configuring and happily
# report a CUDA backend that never runs in practice.
import comfy.quant_ops  # noqa: F401

_CK_IMPORT_ERROR = None
try:
    import comfy_kitchen as ck
    from comfy_kitchen.registry import registry as ck_registry
    from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout
except Exception as exc:  # pragma: no cover - depends on the install
    ck = None
    ck_registry = None
    TensorCoreConvRotW4A4Layout = None
    _CK_IMPORT_ERROR = "{}: {}".format(type(exc).__name__, exc)

# 1024x1024 with Krea2's patch size lands on (1024/16)^2 tokens. Sampling shape drives
# every shape-dependent kernel decision (see `_convrot_int4_fused_shared_memory_fits` in
# comfy_kitchen), so probing at the wrong token count can report the wrong path.
_DEFAULT_TOKENS = (1024 // 16) * (1024 // 16)

_FUNC = "convrot_w4a4_linear"


def _is_convrot_w4a4(weight) -> bool:
    """True for a QuantizedTensor carrying this repo's convrot_w4a4 layout."""
    params = getattr(weight, "_params", None)
    if params is None:
        return False
    return hasattr(params, "convrot_groupsize") and hasattr(params, "linear_dtype")


def quantized_linears(diffusion_model):
    """(name, module) for every Linear whose weight is a convrot_w4a4 QuantizedTensor."""
    for name, module in diffusion_model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is not None and _is_convrot_w4a4(weight):
            yield name, module


def branch_buffers(module):
    """(name, tensor) for the low-rank factors attached to a module, in any naming."""
    for name, buf in module.named_buffers(recurse=False):
        if buf is not None and (name.startswith("svdq_l") or name.startswith("_br_l")):
            yield name, buf


def _probe_kwargs(module, tokens: int, device) -> dict:
    """Kwargs shaped exactly like a real `convrot_w4a4_linear` call, for validation.

    The activation and the weight tensors are `torch.empty` rather than copies: the
    registry validates dtype, ndim, device and divisibility, never contents, and a real
    copy of a 6144x16384 packed weight is 50 MB we would rather not move on the card
    that is already OOMing.
    """
    qweight, wscales = TensorCoreConvRotW4A4Layout.get_plain_tensors(module.weight)
    params = module.weight._params
    x_dtype = params.orig_dtype if params.orig_dtype in (
        torch.float32, torch.float16, torch.bfloat16) else torch.bfloat16
    in_features = int(params.orig_shape[1])
    return {
        "x": torch.empty((tokens, in_features), dtype=x_dtype, device=device),
        "qweight": torch.empty(tuple(qweight.shape), dtype=qweight.dtype, device=device),
        "wscales": torch.empty(tuple(wscales.shape), dtype=wscales.dtype, device=device),
        "bias": None,
        "convrot_groupsize": params.convrot_groupsize,
        "quant_group_size": params.quant_group_size,
        "linear_dtype": params.linear_dtype,
    }


def resolve_dispatch(module, tokens: int = _DEFAULT_TOKENS, device=None):
    """Which backend would run this layer? -> (backend, impl_path, failures).

    `failures` maps backend name to why it was skipped, so "ComfyUI disabled cuda" and
    "this shape violates a kernel constraint" do not look alike in a bug report.
    """
    if ck_registry is None:
        return None, None, {"__import__": _CK_IMPORT_ERROR}

    device = device or mm.get_torch_device()
    try:
        kwargs = _probe_kwargs(module, tokens, device)
    except Exception as exc:
        return None, None, {"__probe__": "{}: {}".format(type(exc).__name__, exc)}

    failures = {}
    for name in ("cuda", "triton", "eager"):
        result = ck_registry.validate_backend_for_call(name, _FUNC, kwargs)
        if not result.success:
            failures[name] = "{}: {}".format(result.failed_param, result.failure_reason)

    try:
        backend = ck_registry.get_capable_backend(_FUNC, kwargs)
        impl = ck_registry.get_implementation(_FUNC, kwargs=kwargs)
        impl_path = "{}.{}".format(impl.__module__, impl.__qualname__)
    except Exception as exc:
        return None, None, failures or {"__dispatch__": str(exc)}
    finally:
        del kwargs

    return backend, impl_path, failures


def dispatch_warning(backend: str | None, failures: dict) -> str | None:
    """The message the loader prints when the fast kernel is not in play.

    Deliberately not caveman-terse and not abbreviated: this text ends up pasted into
    issue threads by people who have not read the README.
    """
    if backend == "cuda":
        return None

    cuda_reason = (failures or {}).get("cuda", "unknown")
    lines = [
        "[krea2-svdquant] WARNING: convrot_w4a4 will dispatch to the '{}' backend, not "
        "'cuda'.".format(backend or "<none>"),
        "  The non-CUDA path unpacks int4 to bf16 in Python and runs an ordinary matmul, "
        "so this checkpoint will be SLOWER than fp8 or even plain bf16.",
        "  cuda backend was rejected because: {}".format(cuda_reason),
    ]

    cuda_build = torch.version.cuda
    if cuda_build is None or tuple(int(p) for p in str(cuda_build).split(".")[:1]) < (13,):
        lines.append(
            "  Most likely cause: ComfyUI disables comfy_kitchen's CUDA backend when torch "
            "is built against CUDA < 13 (see comfy/quant_ops.py). You have torch {} "
            "(cuda {}). Install a cu130 or newer torch build.".format(
                torch.__version__, cuda_build)
        )
    lines.append("  Run the 'Krea2 SVDQuant Diagnostics' node (mode=dispatch) for detail.")
    return "\n".join(lines)


def log_dispatch(diffusion_model) -> None:
    """One line on every load, plus a loud block when the fast kernel is not available."""
    try:
        first = next(iter(quantized_linears(diffusion_model)), None)
        if first is None:
            return
        backend, impl_path, failures = resolve_dispatch(first[1])
        if backend is None:
            logging.warning("[krea2-svdquant] could not resolve a convrot_w4a4 backend: %s",
                            failures)
            return
        logging.info("[krea2-svdquant] convrot_w4a4 dispatch backend: %s (%s)",
                     backend, impl_path)
        warning = dispatch_warning(backend, failures)
        if warning:
            logging.warning("%s", warning)
    except Exception:
        # A diagnostic must never be the reason a model fails to load.
        logging.debug("[krea2-svdquant] dispatch probe failed", exc_info=True)


# --------------------------------------------------------------------------- reports


def _comfyui_version() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        out = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"],
                             cwd=root, capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    try:
        import comfyui_version
        return getattr(comfyui_version, "__version__", "unknown")
    except Exception:
        return "unknown"


def report_env(patcher) -> list[str]:
    device = mm.get_torch_device()
    lines = ["== environment =="]
    lines.append("torch                : {}  (cuda build {})".format(
        torch.__version__, torch.version.cuda))
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(device)
        lines.append("device               : {}  (sm_{}{})".format(
            torch.cuda.get_device_name(device), major, minor))
    else:
        lines.append("device               : CUDA not available")
    lines.append("ComfyUI              : {}".format(_comfyui_version()))
    lines.append("comfy_kitchen        : {}".format(
        _CK_IMPORT_ERROR or _package_version("comfy-kitchen")))

    lines.append("")
    lines.append("== memory ==")
    lines.append("vram_state           : {}".format(mm.vram_state))
    lines.append("free / total         : {:.2f} / {:.2f} GiB".format(
        mm.get_free_memory(device) / 1024 ** 3, mm.get_total_memory(device) / 1024 ** 3))
    lines.append("max pinned           : {:.2f} GiB (currently {:.2f} GiB)".format(
        max(0, getattr(mm, "MAX_PINNED_MEMORY", 0)) / 1024 ** 3,
        getattr(mm, "TOTAL_PINNED_MEMORY", 0) / 1024 ** 3))

    lines.append("")
    lines.append("== patcher ==")
    lines.append("class                : {}".format(type(patcher).__name__))
    try:
        lines.append("is_dynamic           : {}".format(patcher.is_dynamic()))
    except Exception:
        lines.append("is_dynamic           : n/a")
    lines.append("model_size           : {:.3f} GiB".format(patcher.model_size() / 1024 ** 3))
    lines.append("loaded_size          : {:.3f} GiB".format(patcher.loaded_size() / 1024 ** 3))
    lines.append("lowvram              : {} ({} patched modules)".format(
        getattr(patcher.model, "model_lowvram", None), patcher.lowvram_patch_counter()))

    diffusion_model = patcher.model.diffusion_model
    devices: dict[str, int] = {}
    branch_bytes = 0
    branch_devices: dict[str, int] = {}
    count = 0
    for _name, module in quantized_linears(diffusion_model):
        count += 1
        key = str(module.weight.device)
        devices[key] = devices.get(key, 0) + 1
        for _bname, buf in branch_buffers(module):
            branch_bytes += buf.numel() * buf.element_size()
            bkey = str(buf.device)
            branch_devices[bkey] = branch_devices.get(bkey, 0) + 1

    lines.append("")
    lines.append("== quantized layers ==")
    lines.append("convrot_w4a4 linears : {}".format(count))
    lines.append("weight devices       : {}".format(devices or "none"))
    lines.append("low-rank factors     : {} tensors, {:.1f} MiB".format(
        sum(branch_devices.values()), branch_bytes / 1024 ** 2))
    lines.append("factor devices       : {}".format(branch_devices or "none"))
    return lines


def _package_version(dist: str) -> str:
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        return "unknown"


def _distinct_shapes(diffusion_model):
    """One representative module per (in, out) shape -- constraints are shape-dependent."""
    seen: dict[tuple, tuple] = {}
    for name, module in quantized_linears(diffusion_model):
        shape = tuple(module.weight._params.orig_shape)
        seen.setdefault(shape, (name, module))
    return seen


def report_dispatch(patcher, tokens: int) -> list[str]:
    lines = ["== comfy_kitchen backends =="]
    if ck_registry is None:
        lines.append("comfy_kitchen unavailable: {}".format(_CK_IMPORT_ERROR))
        return lines

    for name, info in sorted(ck_registry.list_backends().items()):
        lines.append("{:<8} available={:<5} disabled={:<5} reason={}".format(
            name, str(info["available"]), str(info["disabled"]),
            info["unavailable_reason"] or "-"))
        lines.append("         implements {}: {}".format(
            _FUNC, _FUNC in info["capabilities"]))

    lines.append("")
    lines.append("== dispatch per weight shape (tokens={}) ==".format(tokens))
    shapes = _distinct_shapes(patcher.model.diffusion_model)
    if not shapes:
        lines.append("no convrot_w4a4 layers found")
        return lines

    for shape, (name, module) in sorted(shapes.items()):
        backend, impl_path, failures = resolve_dispatch(module, tokens)
        lines.append("{}  [{} -> {}]".format(name, shape[1], shape[0]))
        lines.append("    backend : {}".format(backend or "NONE"))
        lines.append("    impl    : {}".format(impl_path or "-"))
        if backend != "cuda":
            for bname, reason in sorted(failures.items()):
                lines.append("    rejected {}: {}".format(bname, reason))

    backend, _impl, failures = resolve_dispatch(next(iter(shapes.values()))[1], tokens)
    warning = dispatch_warning(backend, failures)
    if warning:
        lines.append("")
        lines.append(warning)
    return lines


def _timed(fn, reps: int = 20, warmup: int = 3) -> float:
    """Milliseconds per call, CUDA-synchronised. Returns nan if the call raises."""
    try:
        for _ in range(warmup):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0 / reps
    except Exception as exc:
        logging.debug("[krea2-svdquant] bench call failed: %s", exc, exc_info=True)
        return float("nan")


def report_bench(patcher, tokens: int) -> list[str]:
    """Per-shape timing of the quantized path against a bf16 reference.

    The number that settles a "no speedup" report is the last column: quant/bf16 above
    1.0 means the quantized kernel is losing to the format it was meant to beat, which
    is only possible on the eager path.
    """
    lines = ["== per-layer benchmark (tokens={}) ==".format(tokens)]
    if not torch.cuda.is_available():
        lines.append("CUDA not available; nothing to measure")
        return lines

    # Timing offloaded weights would measure PCIe, not the kernel.
    mm.load_models_gpu([patcher], force_full_load=True)
    device = mm.get_torch_device()

    lines.append("{:<44} {:>9} {:>9} {:>9} {:>9}".format(
        "layer [in -> out]", "quant ms", "branch ms", "bf16 ms", "quant/bf16"))

    for shape, (name, module) in sorted(_distinct_shapes(patcher.model.diffusion_model).items()):
        out_features, in_features = int(shape[0]), int(shape[1])
        dtype = module.weight._params.orig_dtype
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            dtype = torch.bfloat16
        x = torch.randn((tokens, in_features), dtype=dtype, device=device)

        quant_ms = _timed(lambda: F.linear(x, module.weight))

        factors = [buf for _n, buf in branch_buffers(module)]
        if len(factors) >= 2:
            a1, a2 = factors[0].to(device=device, dtype=dtype), factors[1].to(device=device, dtype=dtype)
            branch_ms = _timed(lambda: F.linear(F.linear(x, a2), a1))
            del a1, a2
        else:
            branch_ms = float("nan")

        try:
            dense = module.weight.dequantize().to(dtype)
            bf16_ms = _timed(lambda: F.linear(x, dense))
            del dense
        except Exception:
            bf16_ms = float("nan")

        ratio = quant_ms / bf16_ms if bf16_ms == bf16_ms and bf16_ms > 0 else float("nan")
        lines.append("{:<44} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.2f}".format(
            "{} [{} -> {}]".format(name, in_features, out_features)[:44],
            quant_ms, branch_ms, bf16_ms, ratio))

        del x
        mm.soft_empty_cache()

    lines.append("")
    lines.append("quant/bf16 < 1.0 means the int4 kernel is winning. Above 1.0 means the "
                 "quantized path is being emulated -- check mode=dispatch.")
    return lines


def report_profile(patcher, tokens: int) -> list[str]:
    """One sampled step under torch.profiler, attributed by CUDA time."""
    lines = ["== profile =="]
    if not torch.cuda.is_available():
        lines.append("CUDA not available; nothing to profile")
        return lines

    mm.load_models_gpu([patcher], force_full_load=True)
    device = mm.get_torch_device()
    shapes = _distinct_shapes(patcher.model.diffusion_model)
    if not shapes:
        lines.append("no convrot_w4a4 layers found")
        return lines

    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=False) as prof:
        for shape, (_name, module) in sorted(shapes.items()):
            dtype = module.weight._params.orig_dtype
            if dtype not in (torch.float16, torch.bfloat16, torch.float32):
                dtype = torch.bfloat16
            x = torch.randn((tokens, int(shape[1])), dtype=dtype, device=device)
            for _ in range(4):
                module(x)
            del x
        torch.cuda.synchronize()

    lines.append(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    return lines


REPORTS = {
    "env": report_env,
    "dispatch": report_dispatch,
    "bench": report_bench,
    "profile": report_profile,
}


def run_report(patcher, mode: str, tokens: int = _DEFAULT_TOKENS) -> str:
    fn = REPORTS[mode]
    lines = fn(patcher) if mode == "env" else fn(patcher, tokens)
    return "\n".join(lines)


class Krea2SVDQuantDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Output of the Krea2 SVDQuant W4A4 Loader."}),
                "mode": (["dispatch", "env", "bench", "profile"], {
                    "tooltip": "dispatch: which kernel actually runs (start here). "
                               "env: versions and memory accounting. "
                               "bench: quantized vs bf16 per layer. "
                               "profile: torch.profiler table.",
                }),
                "tokens": ("INT", {
                    "default": _DEFAULT_TOKENS, "min": 64, "max": 65536, "step": 64,
                    "tooltip": "Sequence length to probe with. 4096 = 1024x1024. Kernel "
                               "selection is shape-dependent, so match your real run.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "advanced/loaders"
    TITLE = "Krea2 SVDQuant Diagnostics"
    DESCRIPTION = ("Passthrough node that reports which comfy_kitchen backend the "
                   "quantized layers dispatch to, plus memory accounting and timings. "
                   "Paste the output into bug reports.")

    def run(self, model, mode, tokens):
        try:
            report = run_report(model, mode, int(tokens))
        except Exception as exc:
            logging.exception("[krea2-svdquant] diagnostics failed")
            report = "diagnostics failed: {}: {}".format(type(exc).__name__, exc)
        logging.info("\n%s", report)
        return {"ui": {"text": [report]}, "result": (model, report)}


NODE_CLASS_MAPPINGS = {"Krea2SVDQuantDiagnostics": Krea2SVDQuantDiagnostics}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2SVDQuantDiagnostics": "Krea2 SVDQuant Diagnostics"}

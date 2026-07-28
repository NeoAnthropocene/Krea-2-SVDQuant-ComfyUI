"""Route float-mask attention calls away from sage-attention.

Sage's kernels take an ``attn_mask``, but a *float* (additive-bias) mask is the one
shape they handle badly:

* ``sageattn_qk_int8_pv_fp16_triton`` stages the mask tile through shared memory on
  top of the K/V pipeline. At head_dim 128 that is 139276 bytes against the 101376
  the SM offers on Ampere/Ada, so the launch dies with
  ``triton.runtime.errors.OutOfResources``.
* the CUDA kernels do launch, but the mask goes through the int8 path and the error
  against a bf16 reference grows ~50x versus the same call with no mask -- over 224
  layers that is a broken image, not a slightly noisy one.

Boolean masks are fine on both, and so is the no-mask case, which is every ordinary
text-to-image step. The only caller that hands Krea2 a float mask is
comfyui-krea2edit's ``ref_boost`` bias, so this guard costs nothing unless that
feature is on, and then it costs sage's speedup on the edit blocks alone.

Installed by the loader on the model it returns, and wrapped at sample time rather
than at load time because ComfyUI's Patch Sage Attention node may well run after us.
"""

from __future__ import annotations

import logging

import torch

import comfy.patcher_extension

_KEY = "krea2_sage_mask_guard"
_warned = False


def _mask_of(args, kwargs):
    """The ``mask`` argument of ``optimized_attention*(q, k, v, heads, mask=...)``."""
    if "mask" in kwargs:
        return kwargs["mask"]
    return args[4] if len(args) > 4 else None


def _guard(override):
    def guarded(func, *args, **kwargs):
        mask = _mask_of(args, kwargs)
        if mask is not None and torch.is_floating_point(mask):
            global _warned
            if not _warned:
                _warned = True
                logging.info("[krea2-svdquant] float attention mask -> stock attention "
                             "(sage mishandles additive-bias masks)")
            return func(*args, **kwargs)
        return override(func, *args, **kwargs)

    guarded.krea2_mask_guard = True
    return guarded


def _wrapper(executor, *args, **kwargs):
    transformer_options = kwargs.get("transformer_options")
    if transformer_options is None:
        for a in reversed(args):
            if isinstance(a, dict):
                transformer_options = a
                break
    if isinstance(transformer_options, dict):
        override = transformer_options.get("optimized_attention_override")
        if override is not None and not getattr(override, "krea2_mask_guard", False):
            transformer_options["optimized_attention_override"] = _guard(override)
    # __call__, not execute() -- execute() re-runs *this* wrapper (see WrapperExecutor).
    return executor(*args, **kwargs)


def install_mask_guard(patcher) -> None:
    transformer_options = patcher.model_options.setdefault("transformer_options", {})
    existing = transformer_options.get("wrappers", {}).get(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {})
    if _KEY in existing:
        return
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _KEY, _wrapper,
        transformer_options)

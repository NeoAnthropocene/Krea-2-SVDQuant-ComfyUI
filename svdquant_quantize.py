"""Run the quantizer from inside ComfyUI instead of a terminal.

Same code path as ``quantize_krea2.py`` on the command line -- this imports `convert` rather
than reimplementing it, so there is one quantizer, not two that drift.

The honest caveats, which the node's DESCRIPTION repeats because people do not read module
docstrings:

* It blocks the queue. 54 s for a single-shot split, ~5.7 min with the refinement loop, on a
  3090. Nothing else runs meanwhile.
* It needs the GPU to itself, so it unloads whatever ComfyUI is holding first. Your next
  generation will pay a reload.
* It writes ~8 GB.
"""

from __future__ import annotations

import logging
import os
import shutil

import comfy.model_management
import comfy.utils
import folder_paths

from .quantize_krea2 import (
    SAMPLER_HINTS,
    convert,
    derive_out_path,
    resolve_format,
)
from .svdquant_diag import _CATEGORY

# A source has to be dequantized to bf16 before it can be requantized, so leave a margin over
# the ~8 GB output rather than exactly it.
_MIN_FREE_BYTES = 12 * 1024 ** 3


def _free_bytes(path: str) -> int:
    return shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free


class Krea2SVDQuantQuantize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_model": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "The BF16 Krea 2 checkpoint to quantize (~24 GB). An "
                               "already-quantized file cannot be used as a source, except "
                               "FP8, which is unpacked back to BF16 first.",
                }),
                "format": (["svdq", "w4a4", "int8", "fp8"], {
                    "default": "svdq",
                    "tooltip": "svdq: 4-bit weights and activations plus a low-rank bf16 "
                               "correction branch. w4a4: the same without the branch - "
                               "smaller and ~9%% faster per step. int8: the most faithful "
                               "option and still ~2x fp8 on Ampere. fp8: storage only.",
                }),
                "rank": ("INT", {
                    "default": 64, "min": 8, "max": 512, "step": 8,
                    "tooltip": "svdq only: size of the low-rank branch. Higher is not "
                               "reliably better - see the accuracy section of the README.",
                }),
                "refine_iters": ("INT", {
                    "default": 0, "min": 0, "max": 200,
                    "tooltip": "svdq only. 0 is a single-shot SVD split (~54s). Higher "
                               "refines the branch against the quantization error and stops "
                               "when it stops paying off (~5.7min at 100). The refinement "
                               "lowers weight reconstruction error; it has not been shown to "
                               "improve the images, so 0 is the default here.",
                }),
                "groupsize": ("INT", {
                    "default": 256, "min": 32, "max": 1024, "step": 32,
                    "tooltip": "convrot rotation group size. Unused for fp8.",
                }),
                "variant": (["turbo", "base", "unknown"], {
                    "default": "unknown",
                    "tooltip": "Which Krea 2 release this is. Affects only the output "
                               "filename and the recorded metadata - quantization is "
                               "identical; what differs is the sampler settings afterwards.",
                }),
                "output_name": ("STRING", {
                    "default": "",
                    "tooltip": "Filename inside models/diffusion_models/. Leave empty to "
                               "derive it from the variant and format.",
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Off means an existing file of the same name is an error "
                               "rather than 8 GB written over your last run.",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    OUTPUT_TOOLTIPS = ("Where the checkpoint was written, and what went into it.",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    TITLE = "Krea2 SVDQuant Quantize"
    DESCRIPTION = ("Builds a quantized Krea 2 checkpoint from a BF16 one, without leaving "
                   "ComfyUI. BLOCKS THE QUEUE while it runs (54s to ~6min), unloads any "
                   "loaded model to free the GPU, and writes ~8 GB. Load the result with the "
                   "Krea2 SVDQuant W4A4 Loader (svdq) or the stock UNETLoader (w4a4/int8/fp8).")

    def run(self, source_model, format, rank, refine_iters, groupsize, variant, output_name,
            overwrite):
        src = folder_paths.get_full_path_or_raise("diffusion_models", source_model)

        # `rank` always carries a value from the widget, so "was it set?" cannot be inferred
        # the way argparse does it. Non-svdq formats simply ignore it here rather than
        # erroring, which is the friendlier reading of a dropdown the user cannot un-set.
        fmt, rank = resolve_format(format, rank, rank_was_set=False)

        if output_name.strip():
            name = output_name.strip()
            if not name.endswith(".safetensors"):
                name += ".safetensors"
            dst = os.path.join(os.path.dirname(src), name)
        else:
            dst, note = derive_out_path(src, format, rank, variant)
            if note:
                logging.info("[krea2-svdquant] %s", note)

        if os.path.exists(dst) and not overwrite:
            raise RuntimeError(
                "{} already exists. Enable 'overwrite', or set a different output_name."
                .format(dst))

        free = _free_bytes(dst)
        if free < _MIN_FREE_BYTES:
            raise RuntimeError(
                "only {:.1f} GB free on the drive holding {}; quantizing needs roughly {:.0f} "
                "GB of headroom.".format(free / 1024 ** 3, os.path.dirname(dst),
                                         _MIN_FREE_BYTES / 1024 ** 3))

        # Quantization wants the card to itself. Without this it competes with whatever the
        # last generation left resident and OOMs on the dequantize-to-bf16 step.
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        pbar = comfy.utils.ProgressBar(1)
        state = {"total": 0}

        def progress(done, total, message):
            if total != state["total"]:
                state["total"] = total
                pbar.total = total
            pbar.update_absolute(done, total)

        logging.info("[krea2-svdquant] quantizing %s -> %s (format %s, rank %s, "
                     "refine_iters %s)", src, dst, fmt, rank, refine_iters if rank else 0)
        summary = convert(src, dst, fmt, groupsize, "cuda", rank, refine_iters,
                          variant=variant, progress_cb=progress)

        hint = SAMPLER_HINTS.get(variant)
        loader = ("Krea2 SVDQuant W4A4 Loader" if rank else "the stock UNETLoader")
        text = "\n".join(x for x in (
            summary, "Load it with {}.".format(loader), hint) if x)
        logging.info("[krea2-svdquant] %s", text)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {"Krea2SVDQuantQuantize": Krea2SVDQuantQuantize}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2SVDQuantQuantize": "Krea2 SVDQuant Quantize"}

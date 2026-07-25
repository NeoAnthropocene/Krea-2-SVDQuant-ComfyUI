# Krea 2 SVDQuant & Native Quantization for ComfyUI

**Model weights, full benchmarks, and all example images:**
[huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI](https://huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI)

This repo holds the ComfyUI custom nodes and the `quantize_krea2.py` conversion script.
The `.safetensors` checkpoints (7.5-8.3 GB each) are hosted on Hugging Face, not here.
Clone this repo into `custom_nodes/`, then download whichever checkpoint you want from
the Hugging Face repo above into `ComfyUI/models/diffusion_models/`.

License: this project modifies Krea 2 and is distributed under the [Krea 2 Community
License Agreement](https://www.krea.ai/krea-2-licensing) - see [LICENSE.md](LICENSE.md).
Not an official Krea product.

Quantized **Krea 2** checkpoints for ComfyUI — about **2x faster** and **a third the
size** of the usual FP8 version, with no calibration dataset needed. Download a
checkpoint, install the custom nodes, load the included workflow, done. Works on both
**Krea 2 Turbo** (distilled, 8 steps) and the **base** release (~50 steps with real CFG);
the conversion is identical for both, only the sampler settings differ.

Works on **any modern NVIDIA GPU** — INT8/W4A4 tensor cores go back to Turing (RTX
20-series and up). Benchmarked on an RTX 3090 (Ampere, sm_86), which is the case most
existing Krea 2 quantization writeups don't cover, since that generation has no FP8 or
NVFP4 tensor cores at all.

> **Requires a cu130 (CUDA 13) or newer PyTorch build.** ComfyUI disables `comfy_kitchen`'s
> CUDA backend entirely on older torch builds, which silently drops every quantized
> checkpoint onto a pure-Python fallback that is *slower than bf16*. If these checkpoints
> are slower than FP8 for you, this is almost certainly why — see
> [Troubleshooting](#troubleshooting).

This is an experimental, built-from-scratch project: the quantization script, the loader
node, and the LoRA node here were all written for this repo against ComfyUI's own
quantization backend. Everything is reproducible — `quantize_krea2.py` regenerates any of
these checkpoints from a BF16 Krea 2 model in 40-100 seconds, or about 6 minutes with the
low-rank refinement pass enabled (the default for `--format svdq`).

This is a **community-produced modification of Krea 2** and is **not an official Krea
product**. Krea 2 is licensed under the [Krea 2 Community License
Agreement](https://www.krea.ai/krea-2-licensing); this repository and its checkpoints are
distributed under the same terms — read them before using these weights, in particular
the revenue threshold on commercial use.

## Quick start

1. **Install the custom nodes.** Open a terminal in your ComfyUI folder and run:
   ```bash
   git clone https://github.com/alperktt/Krea-2-SVDQuant-ComfyUI custom_nodes/krea2-svdquant
   ```
   (No git? Just download this repo as a ZIP and unzip it into `ComfyUI/custom_nodes/`.)
   Restart ComfyUI.

2. **Download one checkpoint** from the *Files* tab of this page (`Krea2-Turbo-...
   .safetensors`, pick one — see the table below) and put it in
   `ComfyUI/models/diffusion_models/`.

3. **Download the text encoder and VAE** (same ones any Krea 2 Turbo workflow needs,
   not specific to this repo):
   - [`qwen3vl_4b_fp8_scaled.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) → `ComfyUI/models/text_encoders/`
   - [`qwen_image_vae.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors) → `ComfyUI/models/vae/`

4. **Load a workflow.** Drag one of the JSON files from the `workflows/` folder here
   into ComfyUI, pick your checkpoint in the loader node, and generate. (These are saved
   in ComfyUI's *API* format — they import and run fine, they just arrive without a
   saved node layout.)

   - `krea2_svdquant_w4a4_t2i.json` → **Turbo**: 8 steps, `cfg 1.0`, zeroed negative.
   - `krea2_base_svdquant_w4a4_t2i.json` → **base**: 50 steps, `cfg 3.5`, real negative
     prompt. Treat those as a starting point and tune them.

   - `Krea2-Turbo-W4A4-noLowRank.safetensors` → use the normal **UNETLoader** node.
   - Any `SVDQuant-W4A4-rank*` checkpoint → use the **Krea2 SVDQuant W4A4 Loader**
     node from this repo instead (it's what shows up after step 1).

That's it. Everything below is background on *why* it's faster and *how accurate* each
option is, for people who want the details.

## Why this exists

The usual advice for making Krea 2 cheaper to run is FP8. That only pays off if your GPU
has FP8 tensor cores — Ada, Hopper, Blackwell. On anything older, FP8 weights get cast
back to bf16 before the matmul and run through cuBLAS, so you save VRAM but gain no
speed. Measured on an RTX 3090, FP8 was *slower* than plain bf16.

The same trap catches weight-only 4-bit quantization (W4A16): if activations stay 16-bit,
the matmul still runs on bf16 tensor cores at bf16 speed. 4-bit weights only reduce
memory bandwidth, which isn't the bottleneck at typical resolutions and batch sizes.

What actually moves the needle is quantizing **activations too**, onto hardware that has
the units for it. **INT8 and W4A4 tensor cores go back to Turing (RTX 20-series)** — far
wider support than FP8. So this repo quantizes Krea 2 Turbo from BF16 straight into
formats ComfyUI already ships native kernels for (`int8_tensorwise` and `convrot_w4a4`
in `comfy_kitchen`), and adds an SVDQuant-style low-rank correction branch on top of the
native W4A4 kernel to claw back accuracy at 4 bits.

No calibration dataset is needed — the `convrot` (group-wise Hadamard rotation) step
spreads outliers analytically, and activations are quantized by the kernel at run time.
Everything here was built from scratch against ComfyUI's own quantization backend.

## Included checkpoints

| file | format | rank | size |
|---|---|---|---|
| `Krea2-Turbo-W4A4-noLowRank.safetensors` | native `convrot_w4a4`, no accuracy branch | - | 7.50 GB |
| `Krea2-Turbo-SVDQuant-W4A4-rank16.safetensors` | `convrot_w4a4` + low-rank branch | 16 | 7.60 GB |
| `Krea2-Turbo-SVDQuant-W4A4-rank64.safetensors` | `convrot_w4a4` + low-rank branch | 64 | 7.90 GB |
| `Krea2-Turbo-SVDQuant-W4A4-rank128.safetensors` | `convrot_w4a4` + low-rank branch | 128 | 8.30 GB |

The no-low-rank file loads with the stock ComfyUI **UNETLoader**. The three `svdq`
checkpoints need the **Krea2 SVDQuant W4A4 Loader** node from this repo (they carry extra
`*.svdq_l1` / `*.svdq_l2` tensors the stock loader doesn't know about).

Higher rank = larger low-rank correction branch = closer to the unquantized model on
paper, but it is **not strictly monotonic in practice** — see the accuracy section below.
Rank 32 and 256 were also produced and benchmarked for accuracy during development but
are not included in this upload; the `quantize_krea2.py` script reproduces them exactly
(`--rank 32` / `--rank 256`) if you want them.

## What's in this repo

| file | what it is |
|---|---|
| `quantize_krea2.py` | Converts a BF16 Krea 2 checkpoint to int8, w4a4, or w4a4 + low-rank (svdq) |
| `svdquant_w4a4.py` | The **Krea2 SVDQuant W4A4 Loader** node — loads `--format svdq` checkpoints (self-contained, no base model needed) |
| `svdquant_lora.py` | The **Krea2 SVDQuant LoRA Loader** node — the stock ComfyUI LoRA loader silently skips the quantized layers on these models |
| `svdquant_diag.py` | The **Krea2 SVDQuant Diagnostics** node — reports which kernel actually runs, plus memory accounting and per-layer timings |
| `diagnose.py` | The same reports from a terminal, without starting ComfyUI |
| `workflows/*.json` | Example ComfyUI workflows (turbo and base) |

Installing this adds three nodes: **Krea2 SVDQuant W4A4 Loader**, **Krea2 SVDQuant LoRA
Loader**, and **Krea2 SVDQuant Diagnostics** — see [Quick start](#quick-start) above.

### Quantize your own checkpoint

```bash
cd ComfyUI/custom_nodes/krea2-svdquant
python quantize_krea2.py /path/to/krea2_bf16.safetensors --format int8
python quantize_krea2.py /path/to/krea2_bf16.safetensors --format w4a4
python quantize_krea2.py /path/to/krea2_bf16.safetensors --format svdq --rank 64
```

Add `--variant turbo` or `--variant base` to get a checkpoint name you'll still recognise
later (`Krea2-Base-SVDQuant-W4A4-rank64.safetensors`) and to record which release it came
from in the file's metadata. It does not change the quantization: the layer selection keys
off block naming, which Turbo and base share, so both produce the same 224-layer split.

Only the 224 transformer-block linears (attention + MLP) are quantized; norms,
modulation, the text-fusion stack, and the final layer stay at full precision — they are
small and disproportionately sensitive to quantization noise. Expect a line like
`quantized 224 layers; 206 tensors passed through; 896 tensors created ...` for either
variant — 224 is the whole target set, and a run that reports **0** quantized layers now
fails loudly with the leaf names it actually found instead of writing a useless file.

An FP8 checkpoint works as a source too — it is reconstructed back to BF16 first. INT8
and W4A4 sources are rejected, since unpacking those needs layer dimensions the file
alone doesn't carry; use the original BF16 (or FP16) release for those.

#### Low-rank refinement

For `--format svdq`, a single SVD of `W` is only a first guess: it finds the directions
that are largest in `W`, which are not the same as the directions the quantizer handles
worst. So the branch is refit against the *current* quantization error and requantized,
repeatedly, keeping the best — the same alternating scheme DeepCompressor uses. On Krea 2
Turbo at rank 64 this cuts reconstruction error by **9.4%**, with all 224 layers
improving.

Because iteration one is exactly the plain single-shot split and the best result is kept,
refining can never do worse. It costs conversion time: roughly **6 minutes** instead of
40-100 seconds. To skip it:

```bash
python quantize_krea2.py model.safetensors --format svdq --rank 64 --refine-iters 0
```

The objective here is weight reconstruction error, which needs no calibration data — it
is the true output error under the assumption that the input covariance is identity, and
spreading outliers with the convrot rotation is what makes that assumption reasonable.
Closing the rest of the gap to DeepCompressor means measuring the real covariance from
sample data, which is what makes their conversions take hours rather than minutes.

## Benchmarks

All numbers measured on an **RTX 3090 24GB**, 1024x1024, 8-step Euler/simple sampling,
`cfg=1.0` (Krea 2 Turbo distilled schedule), from the same BF16 source checkpoint, on a
**cu130 torch build** (see [Troubleshooting](#troubleshooting) — on an older build every
one of these numbers gets worse, and the ordering inverts).

These are Turbo numbers. The base model at ~50 steps with CFG does roughly 12x the
sampling work per image, so the absolute seconds do not transfer; the *ratios* between
formats do, since they come from the same per-layer kernels.

### End to end, per image

Two numbers matter and are easy to conflate: **first run after switching checkpoints**
(pays disk-to-VRAM load time, ~9-15s here) and **warm run** (model already resident,
what you get generating multiple images back to back). ComfyUI's own progress bar
("`8/8 [00:07<00:00, 1.09it/s]`") only covers the KSampler loop; "`Prompt executed in
X seconds`" is CLIP load/encode + model staging + sampling + VAE decode + save combined
— the two numbers can differ by 2x on a cold run.

| checkpoint | size | first run (cold) | warm run | vs. BF16 |
|---|---|---|---|---|
| BF16 (unquantized reference) | 24.48 GB | 25.3 s | 21.3 s | 1.0x |
| FP8 e4m3, scaled (emulated on Ampere) | 12.24 GB | 22.2 s | 19.2 s | 1.1x |
| INT8 tensorwise + convrot (not in this upload) | 13.16 GB | 13.3 s | 10.4 s | 2.0x |
| **W4A4 + convrot, no low-rank branch** | 7.50 GB | 10.3 s | **10.1 s** | 2.1x |
| **W4A4 + SVDQuant low-rank, rank 16/64/128** | 7.6-8.3 GB | ~19.3 s | **10.1-10.2 s** | 2.1x |

Rank does not measurably change warm speed — CLIP text-encode (Qwen3-VL 4B) and VAE
decode overhead dominate a single 1024x1024/8-step/batch-1 image and mask the low-rank
branch's cost. Add a **TorchCompileModel** node (backend `inductor`) after the loader
for a further ~20-25% cut on the sampling portion specifically (see profiling below);
that number does not show up in the table above since it isn't included in this
upload's default workflow.

**FP8 is not faster than BF16 on Ampere** — there are no FP8 tensor cores on this
architecture, so ComfyUI casts to bf16 and calls cuBLAS. It's included here because it's
the most common recommendation online for "quantizing Krea 2," and the numbers show why
that advice doesn't hold on 30-series cards. **INT8 is the fastest *accurate* option**
measured, but is not part of this upload (available via `quantize_krea2.py --format
int8` on your own BF16 checkpoint).

### Per-layer accuracy (cosine similarity / relative error vs. BF16 original)

Measured on real captured activations from a Krea 2 Turbo forward pass (not synthetic
noise), across representative attention and MLP layers:

| format | cosine | relative error | per-layer time |
|---|---|---|---|
| bf16 (reference) | 1.00000 | - | 1.22 - 3.48 ms |
| **int8 + convrot (Hadamard rotation)** | 0.99999 | 0.35 - 0.63% | 0.39 - 1.09 ms |
| int8 per-channel (no rotation) | 0.99993 | 0.45 - 1.47% | 0.35 - 1.01 ms |
| fp8 e4m3, scaled | 0.99996 | 0.39 - 1.28% | 1.95 - 5.14 ms |
| nvfp4 | 0.99968 | 0.74 - 4.00% | 1.49 - 3.93 ms |
| w4a4 + convrot, rank-64 low-rank branch | 0.99933 - 0.99997 | 0.72 - 8.38% | 0.39 - 1.09 ms |
| w4a4 + convrot, no low-rank branch | 0.99569 - 0.99908 | 1.49 - 9.29% | 0.23 - 0.67 ms |

The Hadamard rotation used by `convrot` already does most of what SVDQuant's low-rank
branch does (both are outlier-mitigation strategies), so on top of `convrot_w4a4` the
low-rank branch buys noticeably less than in the original SVDQuant paper — it roughly
halves the error rather than eliminating it. **`int8` is the more accurate choice if
quality matters more than raw speed; `svdq` is the faster, smaller choice.**

### Rank sweep

`--format svdq --rank N` was run for N = 16, 32, 64, 128, 256. Checkpoint sizes:

| rank | size |
|---|---|
| 16 | 7.60 GB |
| 32 | 7.70 GB |
| 64 | 7.90 GB |
| 128 | 8.30 GB |
| 256 | 9.10 GB |

This is an experimental project — the rank sweep is deliberately shipped so people can
try the tradeoff themselves rather than take one number on faith. If you benchmark other
ranks or find a case where one clearly wins, open a discussion on this repo.

### Where the remaining time goes (profiled, `svdq r64`, single denoise step, 175.7 ms)

| component | share |
|---|---|
| W4A4 GEMM (native `comfy_kitchen` cutlass kernel) | 37% |
| elementwise / norm / RoPE / dtype casts | 34% |
| attention (cuDNN flash) | 9% |
| low-rank branch (2 bf16 GEMMs per quantized layer) | 9% |
| W4A4 activation quantization | 8% |

A third of a step is small elementwise kernels, which is why `torch.compile` (backend
`inductor`) helps: add a **TorchCompileModel** node after the loader. Stock ComfyUI
quantized tensors normally break `torch.compile` (Dynamo can't trace into the
`comfy_kitchen` kernel); the W4A4 loader here works around that by marking those calls as
graph breaks so inductor still fuses everything around them. First run after loading pays
~50s of compilation; subsequent runs are warm.

## LoRA

Use **Krea2 SVDQuant LoRA Loader**, not the stock `LoraLoaderModelOnly`. The stock loader
patches `weight += down @ up`, but on these models `.weight` is a `QuantizedTensor` —
patching it that way would mean dequantize → add → requantize, losing the format. In
practice it silently matches only the ~32 non-quantized layers (text-fusion) out of ~256
and misses all 224 transformer-block layers, with no error.

The included loader instead attaches the LoRA as a parallel low-rank branch, which is
mathematically identical for a linear layer (`(W + BA)x == Wx + B(Ax)`) and leaves the
quantized weight untouched. Chain multiple nodes to stack LoRAs. Check the console — it
reports what it matched, e.g. `224 quantized layers, 32 normal layers`.

The branch is installed as a ComfyUI *object patch*, so it belongs to that one model
branch: two LoRA loader nodes hanging off the same checkpoint loader no longer contaminate
each other, and nothing survives past the sampling run. A stack of N LoRAs on one layer is
folded into a single pair of GEMMs rather than N pairs, and LoRA files are cached by
mtime, so changing a strength no longer re-reads them from disk.

## Troubleshooting

Start with the **Krea2 SVDQuant Diagnostics** node (drop it between the loader and the
KSampler, `mode=dispatch`), or from a terminal:

```bash
python diagnose.py --no-load
```

### "It's slower than FP8 / slower than BF16"

Almost always this: **ComfyUI disables `comfy_kitchen`'s CUDA backend when torch was built
against CUDA < 13**, in `comfy/quant_ops.py`:

```python
if cuda_version < (13,):
    ck.registry.disable("cuda")
    logging.warning("WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.")
```

`convrot_w4a4_linear` resolves its backend per call, so with `cuda` disabled it falls
through to the eager implementation — which unpacks int4 to bf16 in Python and runs an
ordinary matmul. That is strictly slower than just running bf16, and the more aggressive
the format the worse it gets. The tell is that the ordering **inverts**: fp8 fastest, int8
middling, w4a4/svdq slowest, the exact opposite of the benchmark table above.

Check with:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

If that prints anything below `13.0`, install a cu130+ torch build. The loader now prints
the resolved backend on every load and shouts if it isn't `cuda`.

### "Pin error." in the console

Harmless. It comes from ComfyUI core (`comfy/model_management.py`), not from this repo,
and means a weight could not be page-locked so a normal (unpinned) host copy was used
instead. Results are identical; you lose a little load/offload bandwidth. Windows caps
locked pages aggressively — `MAX_PINNED_MEMORY` there is 40% of system RAM — so it fires
routinely with a model this size. It is not specific to `svdq`; INT8 checkpoints trigger it
too. The diagnostics node prints your pinned-memory budget under `mode=env`.

### Out of memory on a small card (and int8 works fine)

Fixed. The low-rank factors were attached as non-persistent buffers, which ComfyUI's
`module_size()` — the basis of every VRAM decision, including the lowvram split — could
not see, while `.to(device)` moved them anyway. Worse, the old branch cached its own
device move back onto the module, so once ComfyUI offloaded a layer the factors quietly
came back to the GPU and stayed there, outside all accounting. About 645 MB at rank 64,
which is the difference between fitting and not on an 8 GB card. INT8 checkpoints carry no
branch, so they were never affected.

They are now published into `state_dict()` under their own `svdq_l1` / `svdq_l2` keys and
staged per call via `comfy.model_management.cast_to`, so they are budgeted and offloaded
like any other weight. `mode=env` on the diagnostics node reports the factor devices — under
lowvram they should sit on `cpu` between steps, not `cuda`.

One gap remains and it is upstream, not here: `QuantizedTensor.nbytes` reports only the
packed weight, so the W4A4 `weight_scale` (~3 MB/layer) is still invisible to ComfyUI's
accounting for *any* w4a4 checkpoint, branch or no branch.

### A re-saved checkpoint logs "left over keys in diffusion model"

Expected. Saving the model out of ComfyUI now includes the `svdq_l1` / `svdq_l2` keys, which
is what lets the file round-trip back into this loader — but the stock `UNETLoader` doesn't
know them and says so. Harmless.

## Accuracy vs. the base model, qualitatively

Same seed and prompt against the BF16 reference produces the same composition throughout
this quantization sweep — differences are in surface detail, not structure. Two stress
tests, same seed across all checkpoints:

**Multi-line small text** (a chalkboard menu board with 3 lines of prices) is the harder
case and is where the checkpoints separate:

| checkpoint | result |
|---|---|
| BF16, FP8 | correct |
| INT8 + convrot (not in this upload) | correct |
| W4A4, no low-rank | one digit/word duplicated |
| SVDQuant rank 16 | correct, but a nearby sign's color shifted |
| SVDQuant rank 32 | one line duplicated |
| SVDQuant rank 64 | one digit wrong |
| SVDQuant rank 128 | correct, closest of the SVDQuant series to BF16 |
| SVDQuant rank 256 | two digits swapped |

Rank does not improve monotonically in a single-seed test like this — it reflects
noise sensitivity at that particular seed, not a reliable ranking. **Rank 128 was the
best performer here**, which is part of why it's included in this upload alongside 16
(smallest) and 64 (a common middle ground).

**Large, short text on a curved surface** (2 words on a hand-held cup) was solved by
every checkpoint including W4A4 with no low-rank branch — legible text and object
counts held up across the board; only fine composition details (a person's pose, an
extra utensil) varied, which is normal sampling variance, not a quantization artifact.

**Takeaway:** if your use case is large signage-style text or no text, any checkpoint in
this repo works. If you're rendering dense small text (menus, labels, documents), the
low-rank branch helps but doesn't fully close the gap to INT8/FP8 — reach for
`quantize_krea2.py --format int8` if that's your primary use case.

## Example comparisons

Same seed, same prompt, across all 9 checkpoints tested during development (only 4 are
included in this upload; BF16/FP8/INT8/rank-32/rank-256 are shown for reference since
they're discussed in the benchmarks above).

### Hard case: dense multi-line text

A rainy neon diner sign with a 3-line handwritten chalkboard menu. This is where the
checkpoints visibly separate — see the accuracy table above for the full breakdown.

| BF16 (reference) | INT8 + convrot (not in this upload) |
|---|---|
| ![bf16](examples/neon_sign_text_test/compare_bf16_reference_00001_.png) | ![int8](examples/neon_sign_text_test/compare_int8_convrot_00001_.png) |

| W4A4, no low-rank branch | SVDQuant rank 128 (best of the included ranks) |
|---|---|
| ![w4a4](examples/neon_sign_text_test/compare_w4a4_convrot_nolowrank_00001_.png) | ![r128](examples/neon_sign_text_test/compare_svdq_r128_00001_.png) |

<details>
<summary>All 9 variants for this prompt (BF16, FP8, INT8, W4A4, rank 16/32/64/128/256)</summary>

[`examples/neon_sign_text_test/`](examples/neon_sign_text_test) — file names match the
config names used in the benchmark tables.

</details>

### Easy case: large text, two subjects, low angle

Two people in varied clothing, a low camera angle, and 2 words of large curved text on
a held object. Every checkpoint renders the text correctly here — only fine composition
details vary, which is normal sampling variance, not a quantization artifact.

| BF16 (reference) | SVDQuant rank 64 |
|---|---|
| ![bf16](examples/ice_cream_multisubject_test/cmp2_bf16_reference_00001_.png) | ![r64](examples/ice_cream_multisubject_test/cmp2_svdq_r64_00001_.png) |

<details>
<summary>All 9 variants for this prompt</summary>

[`examples/ice_cream_multisubject_test/`](examples/ice_cream_multisubject_test)

</details>

## Attribution

Krea 2 is developed by [Krea AI](https://www.krea.ai). This repository contains
derivative, modified weights and is licensed under the same [Krea 2 Community License
Agreement](https://www.krea.ai/krea-2-licensing) as the base model. It is a
community contribution, not an official Krea product, and is not endorsed by Krea.

The quantization kernels used here (`int8_tensorwise`, `convrot_w4a4`) are native to
[ComfyUI](https://github.com/comfyanonymous/ComfyUI)'s `comfy_kitchen` backend. The
low-rank branch construction follows the method described in the [SVDQuant
paper](https://arxiv.org/abs/2411.05007) (Li et al., MIT Han Lab), implemented here from
scratch on top of ComfyUI's native kernel rather than the paper's own Nunchaku engine,
which has no
Krea 2 architecture support.

# Krea 2 SVDQuant & Native Quantization for ComfyUI

**Model weights, full benchmarks, and all example images:**
[huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI](https://huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI)

This repo holds the ComfyUI custom nodes and the `quantize_krea2.py` conversion script.
The `.safetensors` checkpoints (7.5-8.3 GB each) are hosted on Hugging Face, not here —
GitHub isn't a great fit for files that size. Clone this repo into `custom_nodes/`, then
download whichever checkpoint you want from the Hugging Face repo above into
`ComfyUI/models/diffusion_models/`. Full steps in [Quick start](#quick-start) below.

License: this project modifies Krea 2 and is distributed under the [Krea 2 Community
License Agreement](https://www.krea.ai/krea-2-licensing) — see [LICENSE.md](LICENSE.md).
Not an official Krea product.

Tooling and quantized checkpoints that make **Krea 2 Turbo** run faster and in less VRAM
on ComfyUI. Works on **any modern NVIDIA GPU** (INT8/W4A4 tensor cores go back to the
Turing generation, RTX 20-series and up) — benchmarked here on an RTX 3090 (Ampere,
sm_86), which is the case most existing Krea 2 quantization writeups don't cover, since
that generation has no FP8 or NVFP4 tensor cores.

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
   into ComfyUI, pick your checkpoint in the loader node, and generate.

   - `Krea2-Turbo-W4A4-noLowRank.safetensors` → use the normal **UNETLoader** node.
   - Any `SVDQuant-W4A4-rank*` checkpoint → use the **Krea2 SVDQuant W4A4 Loader**
     node from this repo instead (it's what shows up after step 1).

That's it. Everything below is background on *why* it's faster and *how accurate* each
option is, for people who want the details.

## Why this exists

The officially released Krea 2 SVDQuant checkpoint (`transformer_svdquant.safetensors`,
via `Patil/krea-turbo-svdquant`) is **W4A16**: weights are 4-bit, activations stay
16-bit. On Ampere that means the matmuls still run on the same bf16 tensor cores as an
unquantized model — the 4-bit weights only save memory bandwidth, which isn't the
bottleneck at typical batch/token sizes. Worse, Ampere has no FP8 tensor cores either, so
the common "just use FP8" advice also just gets cast to bf16 and run through cuBLAS.

The actual bottleneck-breaking hardware Ampere *does* have is **native INT8 tensor
cores**. This repo quantizes Krea 2 Turbo directly to formats ComfyUI already has native
kernels for (`int8_tensorwise` and `convrot_w4a4`, both in `comfy_kitchen`), and adds
back an SVDQuant-style low-rank branch on top of the native W4A4 kernel for the cases
where you want quantized *activations* without giving up the SVDQuant accuracy trick.

No calibration dataset is needed for any of this — the `convrot` (group-wise Hadamard
rotation) step spreads outliers analytically, and activations are quantized by the
kernel at run time.

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
| `quantize_krea2.py` | Converts a BF16 Krea 2 checkpoint to int8, w4a4, or w4a4+low-rank (svdq) |
| `svdquant_loader.py` | Loads the *officially released* W4A16 SVDQuant checkpoint (merges it with a base model, since that checkpoint ships transformer-linears only) |
| `svdquant_lora.py` | LoRA loader that works on both the W4A16 and W4A4 quantized models — the stock ComfyUI LoRA loader silently patches only ~12% of layers on these models |
| `svdquant_w4a4.py` | Loads checkpoints produced by `quantize_krea2.py --format svdq` (self-contained, no base model needed) |
| `fast_kernel.py` | A bf16 tensor-core Triton W4A16 kernel, ~3.5x faster than the kernel shipped with the official checkpoint on Ampere |
| `extract_base_parts.py` | Strips a full checkpoint down to just the ~0.92 GB the W4A16 loader actually needs |
| `workflows/*.json` | Example ComfyUI workflows |

Installing this adds four nodes: **Krea2 SVDQuant Loader**, **Krea2 SVDQuant LoRA
Loader**, **Krea2 SVDQuant W4A4 Loader** — see [Quick start](#quick-start) above.

### Quantize your own checkpoint

```bash
cd ComfyUI/custom_nodes/krea2-svdquant
python quantize_krea2.py /path/to/krea2_turbo_bf16.safetensors --format int8
python quantize_krea2.py /path/to/krea2_turbo_bf16.safetensors --format w4a4
python quantize_krea2.py /path/to/krea2_turbo_bf16.safetensors --format svdq --rank 64
```

Only the 224 transformer-block linears (attention + MLP) are quantized; norms,
modulation, the text-fusion stack, and the final layer stay at full precision — they are
small and disproportionately sensitive to quantization noise. Runtime per conversion is
roughly 40-100 seconds on an RTX 3090.

## Benchmarks

All numbers measured on an **RTX 3090 24GB**, 1024x1024, 8-step Euler/simple sampling,
`cfg=1.0` (Krea 2 Turbo distilled schedule), from the same BF16 source checkpoint.

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

Full accuracy/speed benchmarking across the sweep is in progress — the r64 numbers above
are the fully measured baseline. If you test other ranks, results and PRs are welcome.

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

Both loader nodes are paired with **Krea2 SVDQuant LoRA Loader** — do not use the stock
`LoraLoaderModelOnly` on these models. It patches `weight += down @ up`, but the
quantized layers either have no plain `.weight` (W4A16) or a `QuantizedTensor` that can't
be patched that way without breaking the quantization (W4A4). The stock loader ends up
silently matching only the ~32 non-quantized layers (text-fusion) out of ~256 and misses
all 224 transformer-block layers with no error. The included loader instead attaches the
LoRA as a parallel low-rank branch, which is mathematically identical for a linear layer
(`(W + BA)x == Wx + B(Ax)`) and keeps the quantized weight untouched.

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
paper](https://arxiv.org/abs/2411.05007) (Li et al., MIT Han Lab), reimplemented here on
top of ComfyUI's native kernel rather than the paper's own Nunchaku engine, which has no
Krea 2 architecture support.

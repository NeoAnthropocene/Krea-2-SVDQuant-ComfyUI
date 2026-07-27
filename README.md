# Krea 2 SVDQuant & Native Quantization for ComfyUI

**Model weights, full benchmarks, and all example images:**
[huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI](https://huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI)

This repo holds the ComfyUI custom nodes and the `quantize_krea2.py` conversion script.
The `.safetensors` checkpoints (7.5-9.1 GB each) are hosted on Hugging Face, not here.
Clone this repo into `custom_nodes/`, then download whichever checkpoint you want from
the Hugging Face repo above into `ComfyUI/models/diffusion_models/`.

Quantized **Krea 2** checkpoints for ComfyUI — about **2x faster** and **a third the
size** of the usual FP8 version, with no calibration dataset needed, on both **Krea 2
Turbo** (distilled, 8 steps) and the **base** release (~50 steps with real CFG; the
conversion is identical, only the sampler settings differ). This is an experimental,
built-from-scratch project — the quantization script, loader node, and LoRA node were
all written for this repo against ComfyUI's own quantization backend, and are fully
reproducible (`quantize_krea2.py` regenerates any checkpoint here from a BF16 source in
40-100 seconds, or ~6 minutes with low-rank refinement, the default for `--format svdq`).

Works on **any modern NVIDIA GPU** — INT8/W4A4 tensor cores go back to Turing (RTX
20-series and up). Benchmarked on an RTX 3090 (Ampere, sm_86), which is the case most
existing Krea 2 quantization writeups don't cover, since that generation has no FP8 or
NVFP4 tensor cores at all.

> **Requires a cu130 (CUDA 13) or newer PyTorch build.** ComfyUI disables `comfy_kitchen`'s
> CUDA backend entirely on older torch builds, which silently drops every quantized
> checkpoint onto a pure-Python fallback that is *slower than bf16*. If these checkpoints
> are slower than FP8 for you, this is almost certainly why — see
> [Troubleshooting](#troubleshooting).

This is a community-produced modification of Krea 2, not an official Krea product —
license and attribution details are at the [bottom of this README](#attribution); read
them before using these weights, in particular the revenue threshold on commercial use.

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

4. **Load a workflow.** Drag one of these from the `workflows/` folder into ComfyUI, pick
   your checkpoint in the loader node, and generate. Each one opens with a **READ ME FIRST**
   note covering the settings that matter.

   - `krea2_turbo_svdquant_w4a4_t2i.json` → **Turbo**: 8 steps, `cfg 1.0`, zeroed negative.
   - `krea2_base_svdquant_w4a4_t2i.json` → **base**: 50 steps, `cfg 3.5`, real negative
     prompt. Treat those as a starting point and tune them.

   The matching `*_api.json` files are for POSTing to `/prompt` from a script — don't drag
   those in, they carry no layout.

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

Higher rank = larger low-rank correction branch = lower weight reconstruction error. Over
four sampled layers: 0.127 at rank 16, 0.098 at rank 64, 0.080 at rank 128. All are built
with `refine_iters=100`, which is what makes rank worth spending at all.

**Which one to download.** Lower reconstruction error stops translating into visibly closer
*images* past rank 64 — unless you load a LoRA, and then it keeps paying up to 256. Measured
paired over 16 prompts x 2 seeds; details and the numbers are in
[BENCHMARKS.md](BENCHMARKS.md#test-3--paired-lpips-fidelity-with-and-without-a-lora).

| you | pick |
|---|---|
| never use LoRAs | **rank 64** — 128 and 256 measure the same, so the extra GB is wasted |
| use LoRAs | **rank 256** — rank 64 loses most of its advantage under one |
| want the smallest/fastest and can accept the drop | no-low-rank, or rank 16 |

Each file records how it was built in its safetensors metadata (`krea2_svdquant_rank`,
`krea2_svdquant_refine_iters`, tool version, source file), so you can check what you
downloaded rather than trusting this table:

```python
from safetensors import safe_open
with safe_open("Krea2-Turbo-SVDQuant-W4A4-rank64.safetensors", framework="pt") as f:
    print(f.metadata())
```

> The test that matters is `f.metadata() is None`, not the date: an early batch (published
> before 2026-07-26) was built without refinement and carries no metadata at all. If yours
> returns `None`, re-download — at rank 128 the unrefined build measures 0.095 against the
> refined 0.080, and the whole rank ladder is flat without refinement.

Rank 32 and 256 were also produced and benchmarked during development but are not included
in this upload; `quantize_krea2.py` reproduces them exactly (`--rank 32` / `--rank 256`).

## What's in this repo

| file | what it is |
|---|---|
| `quantize_krea2.py` | Converts a BF16 Krea 2 checkpoint to int8, w4a4, or w4a4 + low-rank (svdq) |
| `svdquant_w4a4.py` | The **Krea2 SVDQuant W4A4 Loader** node — loads `--format svdq` checkpoints (self-contained, no base model needed) |
| `svdquant_lora.py` | The **Krea2 SVDQuant LoRA Loader** node — applies LoRAs as a parallel branch so the 4-bit weight is never dequantized |
| `svdquant_quantize.py` | The **Krea2 SVDQuant Quantize** node — the quantizer above, run from inside ComfyUI instead of a terminal |
| `svdquant_capture.py` | The **Krea2 SVDQuant Capture Start/Save** nodes — record per-channel activation RMS for `--act-stats` |
| `svdquant_diag.py` | The **Krea2 SVDQuant Diagnostics** and **Krea2 SVDQuant Env Check** nodes — which kernel actually runs, plus memory accounting and per-layer timings |
| `diagnose.py` | The same reports from a terminal, without starting ComfyUI |
| `tools/build_workflows.py` | Regenerates `workflows/*.json`. Edit this, not the JSON |
| `tools/pixel_metrics.py` | LPIPS/PSNR/SSIM against a BF16 reference — see [Benchmarks](#benchmarks) |
| `tools/fidelity_bench.py` | Paired multi-seed multi-LoRA LPIPS harness — every fidelity claim here comes from it |
| `tools/contact_sheet.py` | Per-prompt contact sheets (checkpoints x seeds) for looking at, not scoring |
| `workflows/*.json` | Example workflows — see the format note below |

Installing this adds seven nodes, all under the **Krea2/SVDQuant** category:

| node | what it is for |
|---|---|
| **Krea2 SVDQuant W4A4 Loader** | Loads an `svdq` checkpoint. Its `status` output names the kernel that will actually run — read it first if generation is slow |
| **Krea2 SVDQuant LoRA Loader** | LoRAs and LoKrs on quantized blocks |
| **Krea2 SVDQuant Quantize** | Builds a quantized checkpoint without leaving ComfyUI. Blocks the queue while it runs (54 s to ~6 min) and writes ~8 GB |
| **Krea2 SVDQuant Diagnostics** | Backend dispatch, memory accounting, per-layer timings, profiler table |
| **Krea2 SVDQuant Env Check** | Is the int4 kernel available at all? Needs no model, so you can ask before downloading 8 GB |
| **Krea2 SVDQuant Capture Start** / **Capture Save** | Record activation statistics for an activation-aware build — see [`--act-stats`](#activation-aware-branch---act-stats) |

### Two workflow formats, and why

ComfyUI has two JSON dialects and mixing them up is a bad first five minutes:

- `workflows/krea2_*_t2i.json` — **UI format.** Drag these into the ComfyUI canvas. They
  carry layout, node titles, colours, and a **READ ME FIRST** note with the settings that
  matter and the slow-generation checklist.
- `workflows/krea2_*_t2i_api.json` — **API format.** What you POST to `/prompt` from a
  script. No layout; dragging one in gives you a pile of untitled nodes.

Regenerate the UI ones with `python tools/build_workflows.py` rather than editing the JSON.

### Quantize your own checkpoint

Either from a terminal:

```bash
cd ComfyUI/custom_nodes/krea2-svdquant
python quantize_krea2.py /path/to/krea2_bf16.safetensors --format int8
python quantize_krea2.py /path/to/krea2_bf16.safetensors --format w4a4
python quantize_krea2.py /path/to/krea2_bf16.safetensors --format svdq --rank 64
```

…or with the **Krea2 SVDQuant Quantize** node, which calls the same code with no terminal
involved: drop the source checkpoint in `models/diffusion_models/`, pick it in the node, and
queue. Three things to know before you do:

- **It blocks the queue** for the whole run — 54 s for a single-shot split, ~5.7 min with
  `refine_iters=100`, measured on a 3090. Nothing else generates meanwhile.
- **It takes the GPU.** Any loaded model is unloaded first, so your next generation pays a
  reload.
- **It writes ~8 GB**, and refuses rather than overwriting unless you tick `overwrite`.

**`rank` and `refine_iters` are one lever, not two.** With refinement off, a rank sweep is
flat — rank 16 and rank 256 land within noise of each other (0.337 vs 0.340), so the extra
1.5 GB buys nothing. Raising rank without `refine_iters > 0` is wasted file size; if you want
the cheap build, lower the rank rather than skipping refinement.

**How much rank you need depends on whether you load a LoRA.** Measured paired over 16 prompts
x 2 seeds, refinement on ([BENCHMARKS.md](BENCHMARKS.md#test-3--paired-lpips-fidelity-with-and-without-a-lora)):

| | no LoRA | with a LoRA |
|---|---|---|
| best rank | 64 (128 and 256 tie with it) | **256** |
| r256 vs r64 | t=0.39, wins 16/32 — a coin flip | t=2.8-3.5, wins 23-26/32 |
| r64 vs no branch at all | t=4.45, clearly better | t=0.73-1.72, barely better |

So a rank-64 branch saturates on plain t2i and then largely stops earning its keep once a
LoRA is loaded on top, while rank 256 holds its advantage in both cases. **Rank 256 if you
use LoRAs, rank 64 if you never do.**

<details>
<summary>An earlier version of this section claimed LPIPS falls monotonically with rank — that was wrong</summary>

It read: *"with refinement on, LPIPS falls monotonically with rank across all five ranks
tested (16 → 256), and higher rank helps in 10 of 10 prompts individually."*

That came from 10 prompts at a **single seed**, compared as marginal per-checkpoint means,
and using `wan21-vae` rather than this repo's own `qwen_image_vae`. Re-measured properly —
paired differences, two seeds, 16 prompts, correct VAE — rank 64, 128 and 256 are statistically
indistinguishable without a LoRA (`r128 vs r64` t=0.01 at 16/32). Monotonicity was an artifact
of the weaker measurement, not a property of the checkpoints.

Kept visible rather than quietly deleted: the mistake is the same one the `--rank-alloc`
section below documents, and the fix in both cases was a better statistic, not more data.

</details>

### `--rank-alloc`: where the rank goes, and why it doesn't matter

Rank is uniform across all 224 layers by default, and that is measurably not the efficient
choice. At rank 64, error removed per million branch parameters spans **6.9x** across the eight
projection types — `attn.wk` returns 0.0992 against `mlp.up`'s 0.0143. The cause is GQA: Krea 2
has 12 kv heads against 48 query heads, so `wk`/`wv` are 1536-wide and their branch costs a
third of an MLP branch while absorbing twice as much error.

`--rank-alloc gqa` spends the same bytes accordingly (`wk` 360, `wv` 256, `wq` 72, `wo` 64,
`gate` 56, MLP 8 — 0.02% smaller than uniform rank 64, same speed at 7.1 s/image). **It
does not improve the images** — measured, not assumed; `uniform` stays the default.

<details>
<summary>Why it's kept despite not helping (measured LPIPS, what the greedy solve got wrong)</summary>

LPIPS 0.3523 against uniform's 0.3403, better on 5 of 10 prompts, paired t = +0.55 on 9 df
— no effect in either direction. The greedy solve predicted 6% less weight error and that
did not translate. It does halve the spread across prompts (variance ratio 4.64, F-test
p = 0.032) and improve the worst prompt, 0.4975 → 0.4470, which is worth someone
re-testing at more than 10 images but is not a reason to change the default.

Kept because the mechanism is sound and the option is cheap to leave in. The transferable
result is negative: weight reconstruction error is a poor predictor of image outcome on
this model — three separate attempts to optimise against it (the refinement objective,
per-block depth allocation, this) have failed to move LPIPS in the predicted direction.

</details>

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

<details>
<summary>What the objective is, and the remaining gap to DeepCompressor</summary>

The default objective here is weight reconstruction error, which needs no calibration data
— it is the true output error under the assumption that the input covariance is identity,
and spreading outliers with the convrot rotation is what makes that assumption reasonable.
`--act-stats` (below) relaxes that assumption to a measured per-channel diagonal; the full
covariance, which is what makes DeepCompressor's conversions take hours, is still not
modelled.

</details>

#### Activation-aware branch (`--act-stats`)

Identity input covariance is an approximation, and a measurable one. Capture the real
per-input-channel activation RMS on a calibration pass, then fit the branch against
`||(W - (Q + L1 L2)) * d||_F` instead — the branch spends its rank where the activations
actually are. `d` is normalised to mean 1 and floor-clamped at 0.05 so a near-dead channel
cannot dominate the fit.

**This costs nothing at inference.** Same shapes, same format, same kernels — only the
values inside `svdq_l1`/`svdq_l2` change.

Two nodes capture the statistics (they hook all 224 branched linears on a BF16 model):

1. **Krea2 SVDQuant Capture Start** — between your model loader and the sampler.
2. **Krea2 SVDQuant Capture Save** — takes the sampler's `LATENT` so it runs after
   denoising; set `keep_capturing` on every prompt but the last so several prompts
   accumulate into one file. Writes to `models/krea2_act_stats/`.

Use prompts that are *not* the ones you plan to judge the checkpoint with, then:

```bash
python quantize_krea2.py model.safetensors --format svdq --rank 256 \
  --act-stats krea2_act_stats.safetensors
```

The output filename gains an `-actaware` tag and the file records which stats built it in
`krea2_svdquant_act_stats`. Missing stats for any branched layer is a hard error rather
than a silent fallback. Measured effect: without a LoRA, LPIPS to BF16 drops 0.3378 to
**0.2825** (t=4.68, 27/32 prompts), beating every other checkpoint in the sweep; with a
LoRA it is neutral across four adapters. See
[Test 4 in BENCHMARKS.md](BENCHMARKS.md#test-4--activation-aware-low-rank-objective).

## Benchmarks

> **Community rank sweep + krea2edit LoRA test:** a full rank-16-through-256 comparison
> (refined and non-refined) across 10 stress-test prompts, plus the same sweep run through
> the [Krea 2 Identity Edit LoRA](https://github.com/lbouaraba/comfyui-krea2edit) on 3 real
> photos (Paris/horse/night edits). Grids, prompts, and speed+quality tables:
> [BENCHMARKS.md](BENCHMARKS.md).

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

To measure it yourself against a BF16 reference: generate matching prompts across
checkpoints into one output folder, then `python tools/pixel_metrics.py --dir
<output-dir>` — it pairs files by name (`bench_<checkpoint>_<prompt>_00001_.png`),
reports LPIPS/PSNR/SSIM per checkpoint, and `--noise-floor` gives you the reseed
distance to judge drift against (see the tool's own docstring for details).

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
patches `weight += down @ up`, but on these models `.weight` is a `QuantizedTensor`, so
applying it that way means dequantize → add → requantize: the branch's whole point, keeping
the 4-bit weight untouched, is lost, and the LoRA delta is re-quantized to 4 bits along
with it.

<details>
<summary>Does the stock loader silently skip the quantized layers? Measured: no longer</summary>

This was documented here as "it matches only the ~32 non-quantized layers out of ~256 and
misses all 224 transformer-block layers, with no error." Re-tested against ComfyUI **0.28.0**
(commit `f966a2b3`) on both `W4A4-convrot` and `SVDQuant-W4A4-rank256`, by patching the same
checkpoint at LoRA strength 1.0 and 4.0 and diffing every weight — a layer that was skipped
would be bit-identical between the two:

```
key map offers 224/224 of the quantized layers
load_lora produced 256 patches; add_patches accepted 256 keys
strength 1.0 vs 4.0:  quantized 224/224 differ,  plain 32/32 differ
```

All 224 do move, so the coverage claim does not hold on current ComfyUI. We have not
bisected when that changed. The reason to use this repo's node is the requantization above,
not missing coverage.

</details>

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

### "No speedup at all on my RTX 20-series" (Turing)

Different problem, and this one has no fix on our side. The backend resolves to `cuda`, the
kernel runs, nothing is misconfigured — int4 is simply not much faster than int8 on Turing:

- The instruction the fast path is built around, `mma.m16n8k64` (s4×s4→s32), is **Ampere and
  newer**. `comfy_kitchen`'s default convrot path targets `Sm89` and gates that instruction
  behind `__CUDA_ARCH__ >= 800`.
- SM 7.5 is served by separate kernels (`turing_int4.cu`, `turing_int8.cu`) built on a
  smaller MMA tile — `GemmShape<8, 8, 32>` versus the default int8 path's
  `GemmShape<16, 8, 32>` — and without `cp.async` prefetch, which is also SM80+.

Both formats run weaker kernels there, so switching to int8 does not dodge it either.
Rebuilding `comfy_kitchen` yourself will not change it: the SM75 kernels are what you get.

The int4 checkpoint is still worth downloading on these cards for the **smaller VRAM
footprint** — just do not expect the speed column of the benchmark table.

The diagnostics node and `diagnose.py --no-load` now say this explicitly when they detect a
compute-capability-7.x device, so you can tell "my setup is broken" apart from "my card
predates the instruction". Those kernels live in
[comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen), not here — this repo ships no
CUDA build pipeline, and we have no Turing hardware to validate a replacement against, so
the honest answer is to report it upstream rather than have us ship an untested kernel.

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
| ![bf16](examples/ice_cream_multisubject_test/compare_bf16_reference_00001_.png) | ![r64](examples/ice_cream_multisubject_test/compare_svdq_r64_00001_.png) |

<details>
<summary>All 9 variants for this prompt</summary>

[`examples/ice_cream_multisubject_test/`](examples/ice_cream_multisubject_test)

</details>

## Attribution

Krea 2 is developed by [Krea AI](https://www.krea.ai). This repository contains
derivative, modified weights and is licensed under the same [Krea 2 Community License
Agreement](https://www.krea.ai/krea-2-licensing) as the base model — see
[LICENSE.md](LICENSE.md) for the full terms and how they apply to the code here. It is a
community contribution, not an official Krea product, and is not endorsed by Krea.

The quantization kernels used here (`int8_tensorwise`, `convrot_w4a4`) are native to
[ComfyUI](https://github.com/comfyanonymous/ComfyUI)'s `comfy_kitchen` backend. The
low-rank branch construction follows the method described in the [SVDQuant
paper](https://arxiv.org/abs/2411.05007) (Li et al., MIT Han Lab), implemented here from
scratch on top of ComfyUI's native kernel rather than the paper's own Nunchaku engine,
which has no
Krea 2 architecture support.

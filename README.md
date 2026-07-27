# Krea 2 SVDQuant & Native Quantization for ComfyUI

**Weights and all example images:**
[huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI](https://huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI)

Quantized **Krea 2** checkpoints for ComfyUI — about **2x faster** and **a third the size**
of the usual FP8 version, with no calibration dataset needed, on both **Krea 2 Turbo**
(distilled, 8 steps) and the **base** release. This repo holds the custom nodes and the
`quantize_krea2.py` conversion script; the `.safetensors` files (7.5-9.1 GB each) live on
Hugging Face.

Works on **any modern NVIDIA GPU** — INT8/W4A4 tensor cores go back to Turing (RTX 20-series).
Benchmarked on an RTX 3090 (Ampere), the case most Krea 2 quantization writeups skip, since
that generation has no FP8 or NVFP4 tensor cores at all. Everything here was built from
scratch against ComfyUI's own quantization backend and is reproducible: `quantize_krea2.py`
regenerates any checkpoint from a BF16 source in 40-100 seconds, or ~6 minutes with the
default low-rank refinement.

> **Requires a cu130 (CUDA 13) or newer PyTorch build.** ComfyUI disables `comfy_kitchen`'s
> CUDA backend on older torch builds, which silently drops every quantized checkpoint onto a
> pure-Python fallback that is *slower than bf16*. If these checkpoints are slower than FP8
> for you, this is almost certainly why — see
> [TROUBLESHOOTING.md](TROUBLESHOOTING.md#its-slower-than-fp8--slower-than-bf16).

This is a community-produced modification of Krea 2, not an official Krea product — license
and attribution are at the [bottom of this page](#attribution). Read them before using these
weights, in particular the revenue threshold on commercial use.

## Which checkpoint to download

Everything lives under `checkpoints/` on Hugging Face.

| file | rank | size | when |
|---|---|---|---|
| **`Krea2-Turbo-SVDQuant-W4A4-rank256-actaware.safetensors`** | 256 | 9.10 GB | **Start here.** Closest to BF16 of anything measured, same speed as any other rank-256 build |
| `Krea2-Turbo-SVDQuant-W4A4-rank256.safetensors` | 256 | 9.10 GB | The same without the activation weighting |
| `Krea2-Turbo-SVDQuant-W4A4-rank64.safetensors` | 64 | 7.90 GB | 1.2 GB smaller, and identical to rank 256 *if you never load a LoRA* |
| `Krea2-Turbo-W4A4-noLowRank.safetensors` | — | 7.50 GB | Smallest and ~9% faster per step; loads with the stock **UNETLoader** |
| `checkpoints/legacy/` | 16, 128 | — | Superseded, kept because BENCHMARKS.md cites measurements from them |

The `actaware` file is the same format, size and speed as plain rank 256 — the low-rank branch
was just fitted against measured per-channel activation energy instead of assuming it is
uniform ([`--act-stats`](#activation-aware-branch---act-stats)). Without a LoRA that moves
LPIPS-to-BF16 from 0.3378 to **0.2825** (t=4.68 over 32 paired cells); with a LoRA it measures
no worse. There is no case where the plain build is the better pick. The calibration file it
was built from is in `calibration/`, so the build is reproducible.

**Rank only matters if you load LoRAs.** Without one, ranks 64 / 128 / 256 are statistically
indistinguishable. With one, rank 256 wins clearly and rank 64 loses most of its advantage
over having no branch at all
([Test 3](BENCHMARKS.md#test-3--paired-lpips-fidelity-with-and-without-a-lora)).

Every file records how it was built in its safetensors metadata, so you can check what you
downloaded rather than trusting this table:

```python
from safetensors import safe_open
with safe_open("Krea2-Turbo-SVDQuant-W4A4-rank256-actaware.safetensors", framework="pt") as f:
    print(f.metadata())
```

> The test that matters is `f.metadata() is None`, not the date: an early batch (published
> before 2026-07-26) was built without refinement and carries no metadata at all. If yours
> returns `None`, re-download — the whole rank ladder is flat without refinement.

## Quick start

1. **Install the nodes.** In your ComfyUI folder:
   ```bash
   git clone https://github.com/alperktt/Krea-2-SVDQuant-ComfyUI custom_nodes/krea2-svdquant
   ```
   (No git? Download the repo as a ZIP and unzip it into `ComfyUI/custom_nodes/`.) Restart
   ComfyUI.

2. **Download one checkpoint** from `checkpoints/` on Hugging Face into
   `ComfyUI/models/diffusion_models/`.

3. **Download the text encoder and VAE** — the same ones any Krea 2 Turbo workflow needs:
   - [`qwen3vl_4b_fp8_scaled.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) → `ComfyUI/models/text_encoders/`
   - [`qwen_image_vae.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors) → `ComfyUI/models/vae/`

4. **Drag in a workflow** from `workflows/`. Each opens with a **READ ME FIRST** note covering
   the settings that matter.
   - `krea2_turbo_svdquant_w4a4_t2i.json` → **Turbo**: 8 steps, `cfg 1.0`, zeroed negative.
   - `krea2_base_svdquant_w4a4_t2i.json` → **base**: 50 steps, `cfg 3.5`, real negative prompt.

   Any `SVDQuant-W4A4-rank*` file needs the **Krea2 SVDQuant W4A4 Loader** node from this repo;
   `noLowRank` uses the stock **UNETLoader**. The `*_api.json` files are for POSTing to
   `/prompt` from a script — don't drag those in, they carry no layout.

## See it

Full comparison sheets — 16 prompts x 7 checkpoints x 2 seeds, with and without a LoRA — are
in **[GALLERY.md](GALLERY.md)**. Three of them here. Columns run BF16 reference, no branch,
then rising rank, ending with the activation-aware build; the two rows are two seeds, so
anything that changes between rows is sampling variance rather than the checkpoint.

Dense small text, the hardest case in the set:

![01_dense_text](examples/fidelity_sheets/sheet_base_01_dense_text.jpg)

A portrait — skin texture and catchlights, where quantization is most visible to a viewer:

![11_portrait_woman](examples/fidelity_sheets/sheet_base_11_portrait_woman.jpg)

The same portrait prompt with a realism LoRA loaded on top, BF16 column included:

![11_portrait_woman_lora](examples/fidelity_sheets/sheet_lora2_11_portrait_woman.jpg)

## Why this exists

The usual advice for making Krea 2 cheaper is FP8. That only pays off if your GPU has FP8
tensor cores — Ada, Hopper, Blackwell. On anything older, FP8 weights get cast back to bf16
before the matmul, so you save VRAM and gain no speed; on an RTX 3090, FP8 measured *slower*
than plain bf16. The same trap catches weight-only 4-bit (W4A16): if activations stay 16-bit,
the matmul still runs at bf16 speed.

What moves the needle is quantizing **activations too**, on hardware that has the units for
it — and INT8/W4A4 tensor cores go back to Turing, far wider support than FP8. So this repo
quantizes Krea 2 into formats ComfyUI already ships native kernels for (`int8_tensorwise` and
`convrot_w4a4` in `comfy_kitchen`) and adds an SVDQuant-style low-rank correction branch on
top of the native W4A4 kernel to claw back accuracy at 4 bits.

No calibration dataset is required: the `convrot` group-wise Hadamard rotation spreads
outliers analytically, and the kernel quantizes activations at run time. `--act-stats` can use
a calibration pass if you have one, but the default path needs nothing.

## The nodes

Installing this adds seven nodes, all under **Krea2/SVDQuant**:

| node | what it is for |
|---|---|
| **Krea2 SVDQuant W4A4 Loader** | Loads an `svdq` checkpoint. Its `status` output names the kernel that will actually run — read it first if generation is slow |
| **Krea2 SVDQuant LoRA Loader** | LoRAs and LoKrs on quantized blocks — see [LoRA](#lora) |
| **Krea2 SVDQuant Quantize** | Builds a quantized checkpoint without leaving ComfyUI. Blocks the queue (54 s to ~6 min) and writes ~8 GB |
| **Krea2 SVDQuant Diagnostics** | Backend dispatch, memory accounting, per-layer timings, profiler table |
| **Krea2 SVDQuant Env Check** | Is the int4 kernel available at all? Needs no model, so you can ask before downloading 8 GB |
| **Krea2 SVDQuant Capture Start** / **Capture Save** | Record activation statistics for an activation-aware build |

<details>
<summary>Which file is which</summary>

| file | what it is |
|---|---|
| `quantize_krea2.py` | Converts a BF16 Krea 2 checkpoint to int8, w4a4, or w4a4 + low-rank (svdq). Also the CLI |
| `svdquant_w4a4.py` | The **W4A4 Loader** node — loads `--format svdq` checkpoints (self-contained, no base model needed) |
| `svdquant_lora.py` | The **LoRA Loader** node — applies LoRAs as a parallel branch so the 4-bit weight is never dequantized |
| `svdquant_quantize.py` | The **Quantize** node — the quantizer above, from inside ComfyUI |
| `svdquant_capture.py` | The **Capture Start/Save** nodes — record per-channel activation RMS for `--act-stats` |
| `svdquant_diag.py` | The **Diagnostics** and **Env Check** nodes |
| `diagnose.py` | The same reports from a terminal, without starting ComfyUI (`--mode all` for everything) |
| `tools/fidelity_bench.py` | Paired multi-seed multi-LoRA LPIPS harness. Every fidelity claim in this repo comes from it |
| `tools/contact_sheet.py` | Builds the contact sheets in [GALLERY.md](GALLERY.md) — for looking at, not scoring |
| `tools/pixel_metrics.py` | The shared LPIPS/PSNR/SSIM implementation `fidelity_bench` imports. Not a CLI |
| `tools/build_workflows.py` | Regenerates `workflows/*.json`, both dialects. Edit this, not the JSON |

ComfyUI has two JSON dialects and mixing them up is a bad first five minutes: the plain names
are **UI format** (drag into the canvas — layout, titles, the READ ME note), and `*_api.json`
is **API format** (POST to `/prompt`, no layout). Both come out of `build_workflows.py`.

</details>

## Build your own checkpoint

From a terminal:

```bash
cd ComfyUI/custom_nodes/krea2-svdquant
python quantize_krea2.py /path/to/krea2_bf16.safetensors --format svdq --rank 256 \
  --act-stats krea2_act_stats.safetensors
```

`--format int8` and `--format w4a4` also work (`int8` is the most faithful option and still
~2x FP8 on Ampere; `w4a4` is the fastest and smallest). Add `--variant turbo` or
`--variant base` for a filename you will recognise later and to record the source release in
the metadata — it does not change the quantization, since layer selection keys off block
naming that Turbo and base share.

Or use the **Krea2 SVDQuant Quantize** node, which calls the same code. Three things first: it
blocks the queue for the whole run, it unloads any resident model to take the GPU, and it
writes ~8 GB (refusing rather than overwriting unless you tick `overwrite`).

**`rank` and `refine_iters` are one lever, not two.** With refinement off a rank sweep is flat
— rank 16 and rank 256 land within noise of each other — so raising rank without
`refine_iters > 0` is wasted file size. If you want a cheap build, lower the rank instead of
skipping refinement.

Only the 224 transformer-block linears (attention + MLP) are quantized; norms, modulation, the
text-fusion stack and the final layer stay at full precision, being small and
disproportionately sensitive. Expect `quantized 224 layers; 206 tensors passed through; ...`
— a run reporting **0** fails loudly with the leaf names it actually found rather than writing
a useless file. An FP8 source works too (it is reconstructed to BF16 first); INT8 and W4A4
sources are rejected, because unpacking those needs layer dimensions the file alone lacks.

### Activation-aware branch (`--act-stats`)

The branch is normally fitted by minimising `||W - (Q + L1 L2)||`, which treats every input
channel as carrying the same activation energy. It does not. `--act-stats` weights the
objective by measured per-input-channel activation RMS instead, so the branch spends its rank
where the activations actually are. `d` is normalised to mean 1 and floor-clamped at 0.05 so a
near-dead channel cannot dominate the fit.

**This costs nothing at inference** — same shapes, same format, same kernels; only the values
inside `svdq_l1`/`svdq_l2` change. It is the largest fidelity gain measured in this repo
([Test 4](BENCHMARKS.md#test-4--activation-aware-low-rank-objective)), so it is worth the
extra calibration pass whether or not you use LoRAs.

Two nodes capture the statistics (they hook all 224 branched linears on a BF16 model):

1. **Krea2 SVDQuant Capture Start** — between your model loader and the sampler.
2. **Krea2 SVDQuant Capture Save** — takes the sampler's `LATENT`, so it runs after denoising.
   Leave `keep_capturing` on for every prompt but the last, and several prompts accumulate
   into one file under `ComfyUI/output/`.

Use prompts that are *not* the ones you plan to judge the checkpoint with, then pass the file
to `--act-stats` (or to the Quantize node's `act_stats` input). The output filename gains an
`-actaware` tag and records which statistics built it in `krea2_svdquant_act_stats`. Stats
missing for any branched layer is a hard error, checked before any GPU work rather than
silently falling back.

<details>
<summary>Low-rank refinement, and what the objective actually is</summary>

For `--format svdq`, a single SVD of `W` is only a first guess: it finds the directions
largest in `W`, which are not the directions the quantizer handles worst. So the branch is
refit against the *current* quantization error and requantized, repeatedly, keeping the best —
the alternating scheme DeepCompressor uses. On Krea 2 Turbo at rank 64 this cuts
reconstruction error by **9.4%**, with all 224 layers improving. Because iteration one is
exactly the plain single-shot split and the best result is kept, refining can never do worse.
It costs conversion time: ~6 minutes instead of 40-100 seconds. `--refine-iters 0` skips it.

The default objective is weight reconstruction error, which needs no calibration data — it is
the true output error under the assumption that the input covariance is identity, and
spreading outliers with the convrot rotation is what makes that assumption reasonable.
`--act-stats` relaxes it to a measured per-channel diagonal. The full covariance, which is
what makes DeepCompressor's conversions take hours, is still not modelled.

</details>

<details>
<summary><code>--rank-alloc</code>: where the rank goes, and why it does not matter</summary>

Rank is uniform across all 224 layers by default, and that is measurably not the efficient
choice. At rank 64, error removed per million branch parameters spans **6.9x** across the
eight projection types — `attn.wk` returns 0.0992 against `mlp.up`'s 0.0143. The cause is GQA:
Krea 2 has 12 kv heads against 48 query heads, so `wk`/`wv` are 1536-wide and their branch
costs a third of an MLP branch while absorbing twice the error.

`--rank-alloc gqa` spends the same bytes accordingly (`wk` 360, `wv` 256, `wq` 72, `wo` 64,
`gate` 56, MLP 8 — 0.02% smaller than uniform rank 64, same speed). **It does not improve the
images:** LPIPS 0.3523 against uniform's 0.3403, better on 5 of 10 prompts, paired t = +0.55 —
no effect in either direction. It does halve the spread across prompts (variance ratio 4.64,
F-test p = 0.032) and improve the worst prompt, 0.4975 → 0.4470, which someone could re-test
at more than 10 images. `uniform` stays the default.

Kept because the mechanism is sound and the option is cheap to leave in. The transferable
result is negative: weight reconstruction error is a poor predictor of image outcome on this
model. Three attempts to optimise against it — the refinement objective, per-block depth
allocation, and this — failed to move LPIPS in the predicted direction. `--act-stats` is the
one that worked, and it optimises against something else.

</details>

## What was measured

Full tables, methodology and caveats: **[BENCHMARKS.md](BENCHMARKS.md)**. Images:
**[GALLERY.md](GALLERY.md)**. In short, on an RTX 3090 at 1024x1024 / 8 steps / cfg 1.0:

* **Speed.** W4A4 with no branch is ~2.9x BF16; the low-rank branch costs ~9-10% of step time
  and barely varies with rank, so rank 256 is only ~4% slower than rank 16. FP8 is *slower*
  than BF16 on Ampere. INT8 is the fastest accurate option.
* **Fidelity.** Every claim comes from `tools/fidelity_bench.py`: 16 prompts x 2 seeds x 5 LoRA
  arms, LPIPS against a BF16 reference generated in the *same* arm, compared as paired
  differences with a reseed noise floor printed beside every result. Two findings:
  **branch rank only matters under a LoRA**, and **the activation-aware objective beats every
  other checkpoint here without one**.
* **What was withdrawn.** An earlier quality ranking came from an LLM judge that had
  saturated (9 of 12 rows a flat 10.00/10) and a rank claim came from one seed compared as
  marginal means. Both are struck through in place with the reason, rather than deleted.

## LoRA

Use **Krea2 SVDQuant LoRA Loader**, not the stock `LoraLoaderModelOnly`. The stock loader
patches `weight += down @ up`, but on these models `.weight` is a `QuantizedTensor`, so
applying it that way means dequantize → add → requantize: the branch's whole point, keeping
the 4-bit weight untouched, is lost, and the LoRA delta gets quantized to 4 bits along with it.

This loader instead attaches the LoRA as a parallel low-rank branch, which is mathematically
identical for a linear layer (`(W + BA)x == Wx + B(Ax)`) and leaves the quantized weight alone.
Chain multiple nodes to stack LoRAs; the console reports what matched, e.g. `224 quantized
layers, 32 normal layers`.

The branch is installed as a ComfyUI *object patch*, so it belongs to that one model branch:
two LoRA loader nodes on the same checkpoint loader do not contaminate each other, and nothing
survives past the sampling run. A stack of N LoRAs on one layer folds into a single pair of
GEMMs rather than N pairs, and LoRA files are cached by mtime, so changing a strength does not
re-read them from disk.

<details>
<summary>Does the stock loader silently skip the quantized layers? Measured: not any more</summary>

This page used to claim "it matches only the ~32 non-quantized layers out of ~256 and misses
all 224 transformer-block layers, with no error." Re-tested against ComfyUI **0.28.0** (commit
`f966a2b3`) on both `W4A4-convrot` and `SVDQuant-W4A4-rank256`, by patching the same checkpoint
at LoRA strength 1.0 and 4.0 and diffing every weight — a skipped layer would be bit-identical
between the two:

```
key map offers 224/224 of the quantized layers
load_lora produced 256 patches; add_patches accepted 256 keys
strength 1.0 vs 4.0:  quantized 224/224 differ,  plain 32/32 differ
```

All 224 move, so the coverage claim does not hold on current ComfyUI, and we have not bisected
when that changed. The reason to use this repo's node is the requantization above, not missing
coverage.

</details>

## Troubleshooting

**[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** covers the five things that actually go wrong:
generation slower than FP8 (nearly always a pre-cu130 torch build), no speedup at all on
RTX 20-series, `Pin error.` in the console, out-of-memory on a small card, and "left over keys
in diffusion model" after re-saving.

Start with the **Krea2 SVDQuant Diagnostics** node between the loader and the KSampler, or:

```bash
python diagnose.py --no-load
```

## Attribution

Krea 2 is developed by [Krea AI](https://www.krea.ai). This repository contains derivative,
modified weights and is licensed under the same [Krea 2 Community License
Agreement](https://www.krea.ai/krea-2-licensing) as the base model — see
[LICENSE.md](LICENSE.md) for the full terms and how they apply to the code here. It is a
community contribution, not an official Krea product, and is not endorsed by Krea.

The quantization kernels used here (`int8_tensorwise`, `convrot_w4a4`) are native to
[ComfyUI](https://github.com/comfyanonymous/ComfyUI)'s `comfy_kitchen` backend. The low-rank
branch construction follows the method described in the [SVDQuant
paper](https://arxiv.org/abs/2411.05007) (Li et al., MIT Han Lab), implemented here from
scratch on top of ComfyUI's native kernel rather than the paper's own Nunchaku engine, which
has no Krea 2 architecture support.

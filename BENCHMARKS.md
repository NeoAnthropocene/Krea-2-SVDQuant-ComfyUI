# Rank sweep & krea2edit LoRA benchmarks

Two independent tests on **Krea 2 Turbo**, all checkpoints produced by `quantize_krea2.py`
in this repo. Same seed across every checkpoint in a given test so images are directly
comparable. RTX 3090, cu130 torch build.

> **The quality scores in Tests 1 and 2 have been withdrawn (2026-07-27).** They came from
> an LLM judge (Gemini 3.5 Flash Lite) that turned out to be saturated: 9 of the 12 rows in
> the LoRA table scored a flat 10.00/10, and the whole t2i table spanned 0.46 points. An
> instrument with no discrimination left still produces a confident-looking ranking, so the
> old conclusion — "every SVDQuant rank is statistically indistinguishable" — was an artifact
> of the judge, not a property of the checkpoints. The speed columns are unaffected and kept.
>
> [Test 3](#test-3--paired-lpips-fidelity-with-and-without-a-lora) replaces it with LPIPS
> against a BF16 reference, multi-seed and paired, and reaches the opposite conclusion on the
> question that matters most here: **with a LoRA loaded, branch rank does change fidelity.**

## Test 1 — text-to-image rank sweep

10 prompts picked to stress different failure modes (dense text, hands, crowds, symmetry,
reflections, counting, complex composition), same seed, 1024x1024, 8 steps, cfg 1.0.
Checkpoints: BF16 reference, W4A4 with no low-rank branch, and SVDQuant rank 16/32/64/128/256
in both the fast single-shot split (`--refine-iters 0`, "no refine") and the alternating-refit
version (default, "refined") — see the main README's [Low-rank refinement](README.md#low-rank-refinement)
section for what that flag changes.

[`examples/rank_sweep_t2i_comparison/`](examples/rank_sweep_t2i_comparison/) — one grid PNG
per prompt, all 12 checkpoints side by side with the gemini quality score under each tile.

| id | prompt | stresses |
|---|---|---|
| 01_dense_text | "A rain-soaked neon diner sign at night, below it a handwritten chalkboard menu with three lines of text reading 'SOUP $4 / PIE $6 / COFFEE $2', reflections on wet asphalt, cinematic" | dense multi-line small text |
| 02_curved_text | "Close-up of a person holding a paper coffee cup with large bold curved text 'STAY WARM' printed around the cup, soft morning light, shallow depth of field" | large text on a curved surface |
| 03_hands_detail | "A violinist's hands mid-performance, fingers pressed on the strings, bow in motion with visible blur, studio lighting, extreme close-up, photorealistic" | fine anatomy, hands |
| 04_crowd_faces | "A busy Tokyo street crossing at dusk, dozens of pedestrians with distinct faces and expressions, neon signage in the background, wide angle, high detail" | many small faces |
| 05_symmetry_pattern | "A perfectly symmetrical Islamic geometric tile mosaic, intricate repeating star and polygon pattern, deep blue and gold, overhead flat lighting, ultra sharp" | geometric symmetry / repeating pattern |
| 06_multi_subject | "Two chefs in white uniforms plating a dish together in a busy kitchen, one holding tweezers placing a garnish, the other pouring sauce, steam rising, low angle shot" | multi-subject interaction |
| 07_reflections_glass | "A glass of iced whiskey on a dark wood bar, condensation droplets, warm bokeh lights reflected in the glass and the liquid, macro photography" | specular reflections / transparency |
| 08_logo_typography | "A vintage motorcycle fuel tank with a hand-painted logo reading 'IRON WOLF GARAGE' in bold serif letters, chrome and scratched paint texture, studio product shot" | stylized typography |
| 09_counting_objects | "A wooden table from directly above with exactly seven red apples arranged in a neat row next to three green pears, soft natural light, flat lay photography" | object counting |
| 10_complex_scene | "A fantasy marketplace street at golden hour, merchant stalls with hanging fabrics and baskets of spices, a dragon perched on a rooftop in the background, dense crowd, painterly digital art" | complex composition |

![01_dense_text](examples/rank_sweep_t2i_comparison/grid_01_dense_text.png)
![02_curved_text](examples/rank_sweep_t2i_comparison/grid_02_curved_text.png)
![03_hands_detail](examples/rank_sweep_t2i_comparison/grid_03_hands_detail.png)
![04_crowd_faces](examples/rank_sweep_t2i_comparison/grid_04_crowd_faces.png)
![05_symmetry_pattern](examples/rank_sweep_t2i_comparison/grid_05_symmetry_pattern.png)
![06_multi_subject](examples/rank_sweep_t2i_comparison/grid_06_multi_subject.png)
![07_reflections_glass](examples/rank_sweep_t2i_comparison/grid_07_reflections_glass.png)
![08_logo_typography](examples/rank_sweep_t2i_comparison/grid_08_logo_typography.png)
![09_counting_objects](examples/rank_sweep_t2i_comparison/grid_09_counting_objects.png)
![10_complex_scene](examples/rank_sweep_t2i_comparison/grid_10_complex_scene.png)

## Test 2 — krea2edit LoRA (identity-preserving editing)

Same rank sweep, this time with the [Krea 2 Identity Edit LoRA](https://github.com/lbouaraba/comfyui-krea2edit)
on top, using its `Krea2EditModelPatch` / `Krea2EditGroundedEncode` nodes wired exactly per
the LoRA repo's example workflow (`ref_boost=4`, `fit_mode=fit`, `grounding_px=768`,
10 steps, cfg 1.0). Quantized checkpoints use this repo's **Krea2 SVDQuant LoRA Loader**
node instead of the stock LoRA loader — the stock loader silently skips the quantized
layers on these models (see the main README's [LoRA](README.md#lora) section).

Three real stock photos of different women, resized to 1024x1536 before editing (feeding
multi-thousand-pixel originals straight into VAEEncode wastes VRAM/time for no quality
gain at this model's ~1MP working resolution).

| woman 1 | woman 2 | woman 3 |
|---|---|---|
| ![woman1](examples/krea2edit_lora_comparison/source_photos/source_woman1.png) | ![woman2](examples/krea2edit_lora_comparison/source_photos/source_woman2.png) | ![woman3](examples/krea2edit_lora_comparison/source_photos/source_woman3.png) |

| id | source | instruction |
|---|---|---|
| e1_paris_w1 | woman 1 | "Place her in Paris with the Eiffel Tower visible in the background, golden afternoon light, keep her exact face, hair, and outfit unchanged." |
| e2_sunset_sky_w1 | woman 1 | "Change the sky and background to a dramatic sunset with orange and pink clouds, keep her exact face and pose unchanged." |
| e3_horse_w2 | woman 2 | "Show her riding a horse outdoors on a countryside trail, keep her exact face, hat, and outfit unchanged." |
| e4_night_lights_off_w2 | woman 2 | "Change the scene to nighttime, turn off any lights, dark moody night sky, keep her exact face unchanged." |
| e5_paris_w3 | woman 3 | "Place her in Paris with the Eiffel Tower visible behind her, keep her exact face, hairstyle, and outfit unchanged." |
| e6_night_lights_off_w3 | woman 3 | "Change the lighting to nighttime with all lights turned off, dark and moody atmosphere, keep her exact face unchanged." |

![e1_paris_w1](examples/krea2edit_lora_comparison/grid_e1_paris_w1.png)
![e2_sunset_sky_w1](examples/krea2edit_lora_comparison/grid_e2_sunset_sky_w1.png)
![e3_horse_w2](examples/krea2edit_lora_comparison/grid_e3_horse_w2.png)
![e4_night_lights_off_w2](examples/krea2edit_lora_comparison/grid_e4_night_lights_off_w2.png)
![e5_paris_w3](examples/krea2edit_lora_comparison/grid_e5_paris_w3.png)
![e6_night_lights_off_w3](examples/krea2edit_lora_comparison/grid_e6_night_lights_off_w3.png)

## Results

### Speed — T2I rank sweep, 10 prompts, 1024x1024, 8 steps, cfg 1.0

| checkpoint | warm (s) | vs BF16 |
|---|---|---|
| BF16 (reference) | 18.80 | 1.00x |
| W4A4, no low-rank | 6.49 | 2.90x |
| SVDQuant r16, no refine | 7.10 | 2.65x |
| SVDQuant r16, refined | 6.95 | 2.71x |
| SVDQuant r32, no refine | 6.74 | 2.79x |
| SVDQuant r32, refined | 6.86 | 2.74x |
| SVDQuant r64, no refine | 7.16 | 2.63x |
| SVDQuant r64, refined | 7.16 | 2.63x |
| SVDQuant r128, no refine | 7.22 | 2.61x |
| SVDQuant r128, refined | 7.22 | 2.60x |
| SVDQuant r256, no refine | 7.54 | 2.49x |
| SVDQuant r256, refined | 7.77 | 2.42x |

### Speed — krea2edit LoRA, 6 edits, ref_boost=4, 1024x1024, 10 steps, cfg 1.0

| checkpoint | warm (s) | vs BF16 |
|---|---|---|
| BF16 (reference) | 44.86 | 1.00x |
| W4A4, no low-rank | 18.89 | 2.37x |
| SVDQuant r16, no refine | 23.54 | 1.91x |
| SVDQuant r16, refined | 23.96 | 1.87x |
| SVDQuant r32, no refine | 24.04 | 1.87x |
| SVDQuant r32, refined | 23.46 | 1.91x |
| SVDQuant r64, no refine | 23.51 | 1.91x |
| SVDQuant r64, refined | 23.44 | 1.91x |
| SVDQuant r128, no refine | 23.85 | 1.88x |
| SVDQuant r128, refined | 23.87 | 1.88x |
| SVDQuant r256, no refine | 25.09 | 1.79x |
| SVDQuant r256, refined | 25.10 | 1.79x |

One of the six `r16, refined` edit runs hit unrelated GPU contention on the test machine
(extra load from other applications) and was excluded from that row's average as a
measurement artifact, not a property of the checkpoint.

**Speed takeaway:** `W4A4, no low-rank` is the fastest checkpoint in both tests (~2.9x on
plain t2i, ~2.4x on the edit LoRA). The branch costs ~9-10% of step time and barely varies
with rank — rank 256 is only ~4% slower than rank 16.

## Test 3 — paired LPIPS fidelity, with and without a LoRA

Run with [`tools/fidelity_bench.py`](tools/fidelity_bench.py), which exists because the judge
above could not be trusted. **16 prompts x 2 seeds = 32 paired cells per arm**, LPIPS(AlexNet)
against a BF16 reference *generated in the same arm*, 1024x1024 / 8 steps / cfg 1.0 /
euler+simple, `qwen_image_vae`. Three arms:

| arm | LoRA | covers |
|---|---|---|
| `base` | none | — |
| `lora` | `canon_krea2`, rank 16, photographic style | 448 block + 64 txtfusion keys |
| `lora2` | `bloomgirls-ultrarealism`, rank 32, realism style | 448 block + 64 txtfusion keys |

Two rules this test follows and the old one did not:

* **Paired, not marginal.** Prompt difficulty varies far more than checkpoints do (LPIPS
  0.23-0.42 across this set), so comparing two checkpoints' *means* buries the effect. Every
  number below is `mean(A_i - B_i)` over the same prompt+seed cells.
* **Each arm has its own reseed floor**, because a LoRA changes how seed-sensitive the model
  is: 0.5468 (base), 0.5193 (lora), 0.5505 (lora2). Raw LPIPS is therefore **not comparable
  across arms** — only within one. Divide by the arm's own floor to compare across.

### The result: rank saturates at 64 without a LoRA, and much later with one

Mean LPIPS vs BF16 (lower = closer to the unquantized model):

| checkpoint | base | lora | lora2 |
|---|---|---|---|
| W4A4, no low-rank | 0.3954 | 0.3944 | 0.3808 |
| SVDQuant r16 | 0.3758 | 0.3453 | 0.3672 |
| SVDQuant r64 | 0.3325 | 0.3611 | 0.3684 |
| SVDQuant r128 | 0.3324 | 0.3378 | 0.3299 |
| SVDQuant r256 | 0.3378 | **0.2993** | **0.3215** |

Paired `r256 vs r64` (negative = r256 closer to BF16, `***` is |t| > 3):

| arm | mean | t | r256 wins |
|---|---|---|---|
| base | +0.0053 | 0.39 | 16/32 — a coin flip |
| lora | -0.0618 | **3.53*** | 26/32 |
| lora2 | -0.0468 | **2.79** | 23/32 |

Without a LoRA, **r64, r128 and r256 are indistinguishable from each other** (`r128 vs r64`
t=0.01 at 16/32; `r128 vs r256` t=0.31 at 17/32). The branch matters — all three beat
`no low-rank` at t=3.2-5.2 — but past rank 64 it stops buying anything.

Load a LoRA and that ceiling moves. r256 wins clearly in both LoRA arms, and the effect
replicates across two adapters of different rank and different style, and separately across
both halves of the prompt set (objects/scenes n=20, t=2.99; people n=12, t=2.29).

The sharpest way to see it — does the branch beat having no branch at all?

| | base | lora | lora2 |
|---|---|---|---|
| no-low-rank vs r64 | t=4.45*** | t=1.72 | **t=0.73** (12/32) |
| no-low-rank vs r256 | t=5.22*** | t=5.71*** | t=4.15*** |

**Under a LoRA a rank-64 branch is worth close to nothing over no branch at all, while a
rank-256 branch keeps its full advantage.** A sweep run without a LoRA would clear rank 64 as
"enough" — and be wrong for the way most people actually use these checkpoints.

**Recommendation: rank 256 if you use LoRAs, rank 64 if you never do.**

Two honest caveats. First, r64 lands slightly *below* r16 in both LoRA arms, which no simple
"more rank is better" story explains; individually neither comparison is significant (t=0.92
and t=0.06) so it may be noise, but it is consistent and we cannot account for it. Second,
this is two LoRAs at strength 1.0 and two seeds per prompt — enough to show the effect is not
adapter-specific, not enough to map how it scales with LoRA strength or rank.

Ruled out as confounds: r16/r64/r128/r256 all carry `refine_iters=100`, `groupsize=256`, the
same source `turbo.safetensors` and 224 branches each. Rank is the only variable between them.

Raw data: `fb/fidelity_bench.csv` and `fb/fidelity_bench_summary.json` in the ComfyUI output
directory, plus per-prompt contact sheets from
[`tools/contact_sheet.py`](tools/contact_sheet.py) — LPIPS says how far an image moved, not
whether it got worse, so the sheets are there to be looked at rather than trusted.

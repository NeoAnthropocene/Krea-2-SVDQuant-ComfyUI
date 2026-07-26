# Rank sweep & krea2edit LoRA benchmarks

Two independent tests on **Krea 2 Turbo**, all checkpoints produced by `quantize_krea2.py`
in this repo. Same seed across every checkpoint in a given test so images are directly
comparable. RTX 3090, cu130 torch build.

Grading: prompt/instruction fidelity, fine detail, composition match, and freedom from
artifacts were scored 1-10 per axis by an LLM judge (Gemini 3.5 Flash Lite, given the
BF16 output as reference and the quantized output as candidate) and averaged. Treat these
as a rough sanity check, not a rigorous quality ranking — an LLM judge on this few samples
per checkpoint agrees with itself more than it discriminates real quality gaps, and every
checkpoint here scored within about half a point of the others. The images are the actual
evidence; the numbers are a coarse pointer, not a verdict.

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

### T2I quantize rank sweep -- 10 prompts, seed 987654321, 1024x1024, 8 steps, cfg 1.0

| checkpoint | warm (s) | vs BF16 | gemini quality /10 |
|---|---|---|---|
| BF16 (reference) | 18.80 | 1.00x | ref |
| W4A4, no low-rank | 6.49 | 2.90x | 9.60 |
| SVDQuant r16, no refine | 7.10 | 2.65x | 9.50 |
| SVDQuant r16, refined | 6.95 | 2.71x | 9.32 |
| SVDQuant r32, no refine | 6.74 | 2.79x | 9.35 |
| SVDQuant r32, refined | 6.86 | 2.74x | 9.57 |
| SVDQuant r64, no refine | 7.16 | 2.63x | 9.50 |
| SVDQuant r64, refined | 7.16 | 2.63x | 9.78 |
| SVDQuant r128, no refine | 7.22 | 2.61x | 9.65 |
| SVDQuant r128, refined | 7.22 | 2.60x | 9.55 |
| SVDQuant r256, no refine | 7.54 | 2.49x | 9.60 |
| SVDQuant r256, refined | 7.77 | 2.42x | 9.43 |

### krea2edit LoRA -- 6 edits, ref_boost=4, 1024x1024, 10 steps, cfg 1.0

| checkpoint | warm (s) | vs BF16 | gemini quality /10 |
|---|---|---|---|
| BF16 (reference) | 44.86 | 1.00x | ref |
| W4A4, no low-rank | 18.89 | 2.37x | 10.00 |
| SVDQuant r16, no refine | 23.54 | 1.91x | 9.92 |
| SVDQuant r16, refined | 23.96 | 1.87x | 10.00 |
| SVDQuant r32, no refine | 24.04 | 1.87x | 10.00 |
| SVDQuant r32, refined | 23.46 | 1.91x | 10.00 |
| SVDQuant r64, no refine | 23.51 | 1.91x | 9.92 |
| SVDQuant r64, refined | 23.44 | 1.91x | 10.00 |
| SVDQuant r128, no refine | 23.85 | 1.88x | 10.00 |
| SVDQuant r128, refined | 23.87 | 1.88x | 10.00 |
| SVDQuant r256, no refine | 25.09 | 1.79x | 10.00 |
| SVDQuant r256, refined | 25.10 | 1.79x | 10.00 |

One of the six `r16, refined` edit runs hit unrelated GPU contention on the test machine
(extra load from other applications) and was excluded from that row's average as a
measurement artifact, not a property of the checkpoint.

**Takeaway:** `W4A4, no low-rank` is the fastest checkpoint in both tests (~2.9x on plain
t2i, ~2.4x on the edit LoRA) and scores statistically indistinguishable from every SVDQuant
rank on this prompt set. Higher rank does not reliably buy more quality here — same
non-monotonic pattern the main README's rank-16-vs-256 text test shows. If low-rank branch
size matters for your use case, judge it on your own prompts; on this set it mostly buys
peace of mind, not a measurable gain.

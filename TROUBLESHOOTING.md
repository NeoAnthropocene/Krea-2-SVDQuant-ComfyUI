# Troubleshooting

Split out of the README so the getting-started page stays short. Start with the **Krea2
SVDQuant Diagnostics** node (drop it between the loader and the KSampler, `mode=dispatch`),
or from a terminal:

```bash
python diagnose.py --no-load
```

`--mode all` adds the memory, dispatch, benchmark and profile tables in one go, which is
what to paste into a bug report.

## "It's slower than FP8 / slower than BF16"

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

## "No speedup at all on my RTX 20-series" (Turing)

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

## "Pin error." in the console

Harmless. It comes from ComfyUI core (`comfy/model_management.py`), not from this repo,
and means a weight could not be page-locked so a normal (unpinned) host copy was used
instead. Results are identical; you lose a little load/offload bandwidth. Windows caps
locked pages aggressively — `MAX_PINNED_MEMORY` there is 40% of system RAM — so it fires
routinely with a model this size. It is not specific to `svdq`; INT8 checkpoints trigger it
too. The diagnostics node prints your pinned-memory budget under `mode=env`.

## Out of memory on a small card (and int8 works fine)

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

## Sage attention: `OutOfResources` or a broken image with krea2edit's `ref_boost`

Fixed, and it was never about the LoRA. `comfyui-krea2edit` turns `ref_boost` into an
*additive float* attention bias, and a float mask is the one input sage's kernels do not
take well:

* `sageattn_qk_int8_pv_fp16_triton` stages the mask tile through shared memory on top of the
  K/V pipeline. At head_dim 128 it asks for 139276 bytes against the 101376 an Ampere or Ada
  SM offers, so the launch dies with
  `triton.runtime.errors.OutOfResources: out of resource: shared memory`.
* the CUDA kernels launch, but the masked result drifts ~50x further from a BF16 reference
  than the same call unmasked — over 224 layers that is the black or scrambled image.

Boolean masks are fine on both, and so is the ordinary no-mask text-to-image step, which is
why this only ever showed up on the edit workflow. The loader now installs a guard that sends
float-mask attention calls to ComfyUI's stock attention and leaves everything else on sage:
you keep the speedup on every normal block and pay full price only on the biased ones. The
console logs `float attention mask -> stock attention` once per run when it fires.

Merging the LoRA into the checkpoint appeared to fix it only because that comparison also
dropped the `ref_boost` bias; the quantized model and its LoRA branch are not involved. If
you load a plain `--format w4a4` / `int8` checkpoint through the stock `UNETLoader`, the
guard is not installed — use the SVDQuant loader, or turn sage off for edit workflows.

## A re-saved checkpoint logs "left over keys in diffusion model"

Expected. Saving the model out of ComfyUI now includes the `svdq_l1` / `svdq_l2` keys, which
is what lets the file round-trip back into this loader — but the stock `UNETLoader` doesn't
know them and says so. Harmless.

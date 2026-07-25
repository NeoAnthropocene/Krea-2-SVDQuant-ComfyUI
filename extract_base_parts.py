"""Extract the small companion file the SVDQuant checkpoint needs.

The SVDQuant checkpoint only holds the 224 quantized linears. Everything else -
embeddings, norms, modulation, the text-fusion stack, the final layer - still has to
come from a full Krea2 Turbo checkpoint. Those remaining tensors are tiny (~0.7 GB out
of a 12 GB base), so this writes just them to a standalone file.

After running this you can point the loader's ``base_model`` at the companion file and
delete/move the 12 GB base.

    python extract_base_parts.py <base.safetensors> [out.safetensors]
"""

from __future__ import annotations

import json
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file

_QUANT_SUFFIXES = ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo",
                   "mlp.gate", "mlp.up", "mlp.down")
_PARAM_SUFFIXES = (".weight", ".bias", ".scale_weight", ".scale_input",
                   ".weight_scale", ".input_scale", ".comfy_quant")


def _layer_of(name: str) -> str:
    for suffix in _PARAM_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def is_replaced(layer: str) -> bool:
    if not layer.startswith("blocks."):
        return False  # txtfusion blocks stay: the checkpoint does not quantize them
    return any(layer.endswith(s) for s in _QUANT_SUFFIXES)


def extract(base_path: str, out_path: str) -> tuple[int, int, int]:
    tensors = {}
    skipped = 0
    with safe_open(base_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            if is_replaced(_layer_of(key)):
                skipped += 1
                continue
            tensors[key] = handle.get_tensor(key)

    out_meta = {k: v for k, v in metadata.items() if k != "_quantization_metadata"}
    if "_quantization_metadata" in metadata:
        parsed = json.loads(metadata["_quantization_metadata"])
        parsed["layers"] = {k: v for k, v in parsed.get("layers", {}).items()
                            if not is_replaced(k)}
        out_meta["_quantization_metadata"] = json.dumps(parsed)
    out_meta["krea2_svdquant_base_parts"] = "true"
    out_meta["krea2_svdquant_source"] = os.path.basename(base_path)

    save_file(tensors, out_path, metadata=out_meta)
    return len(tensors), skipped, os.path.getsize(out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    base = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(base), "krea2_svdquant_base_parts.safetensors")
    kept, skipped, size = extract(base, out)
    print(f"wrote {out}")
    print(f"  kept {kept} tensors, skipped {skipped} replaced-layer tensors, "
          f"{size / 1024 ** 3:.2f} GB (source was "
          f"{os.path.getsize(base) / 1024 ** 3:.2f} GB)")

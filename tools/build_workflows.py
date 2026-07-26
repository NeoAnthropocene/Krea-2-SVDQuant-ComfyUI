"""Generate the UI-format workflow JSONs under ``workflows/``.

Run this, do not hand-edit the output:

    python tools/build_workflows.py

ComfyUI has two JSON dialects and they are not interchangeable.

* **API format** -- ``{"1": {"class_type": ..., "inputs": {...}}, ...}``. What you POST to
  ``/prompt``. Carries no layout, no titles, no notes, no colours.
* **UI format** -- ``{"nodes": [...], "links": [...], ...}``. What the editor saves and what
  drag-and-drop expects.

The repo used to ship only API format while the README told people to drag the file in, so
every new user's first experience was a pile of untitled nodes stacked on the origin. Both
dialects are emitted now: ``*_api.json`` for scripting, the plain name for the editor.

Positions are computed from a column/row grid rather than written by hand, so adding a node
does not mean renumbering everything below it.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "workflows")

# Widget order matters: ComfyUI matches `widgets_values` positionally against the node's
# required inputs, so these lists follow each node class's INPUT_TYPES order exactly.
COL = 420          # horizontal gap between columns
ROW = 200          # vertical gap between rows

TEAL = ("#233", "#355")        # loaders
GREEN = ("#232", "#353")       # prompts
PURPLE = ("#323", "#535")      # sampling
YELLOW = ("#432", "#653")      # notes / read me first


class Graph:
    """Minimal UI-format graph builder.

    Only what these workflows need: typed slots, links, groups, colours. Not a general
    LiteGraph serializer.
    """

    def __init__(self):
        self.nodes = []
        self.links = []
        self._next_node = 1
        self._next_link = 1
        self._slots = {}   # node_id -> {"in": {name: idx}, "out": {name: idx}}

    def add(self, class_type, col, row, inputs=None, outputs=None, widgets=None,
            title=None, colour=None, size=None, order=None):
        nid = self._next_node
        self._next_node += 1
        inputs = inputs or []
        outputs = outputs or []
        self._slots[nid] = {
            "in": {name: i for i, (name, _) in enumerate(inputs)},
            "out": {name: i for i, (name, _) in enumerate(outputs)},
        }
        node = {
            "id": nid,
            "type": class_type,
            "pos": [60 + col * COL, 60 + row * ROW],
            "size": size or [380, 100],
            "flags": {},
            "order": self._next_node - 2 if order is None else order,
            "mode": 0,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in inputs],
            "outputs": [{"name": n, "type": t, "links": [], "slot_index": i}
                        for i, (n, t) in enumerate(outputs)],
            "properties": {"Node name for S&R": class_type},
            "widgets_values": widgets if widgets is not None else [],
        }
        if title:
            node["title"] = title
        if colour:
            node["color"], node["bgcolor"] = colour
        self.nodes.append(node)
        return nid

    def link(self, src_node, src_name, dst_node, dst_name):
        src_slot = self._slots[src_node]["out"][src_name]
        dst_slot = self._slots[dst_node]["in"][dst_name]
        link_type = next(n for n in self.nodes if n["id"] == src_node)["outputs"][src_slot]["type"]
        lid = self._next_link
        self._next_link += 1
        self.links.append([lid, src_node, src_slot, dst_node, dst_slot, link_type])
        for n in self.nodes:
            if n["id"] == src_node:
                n["outputs"][src_slot]["links"].append(lid)
            elif n["id"] == dst_node:
                n["inputs"][dst_slot]["link"] = lid
        return lid

    def note(self, text, col, row, size=(400, 260), title="READ ME"):
        """A built-in Note node. Its text lives in widgets_values[0]."""
        return self.add("Note", col, row, widgets=[text], title=title, colour=YELLOW,
                        size=list(size))

    def group(self, title, bounding, colour="#3f789e"):
        return {"title": title, "bounding": list(bounding), "color": colour,
                "font_size": 24, "flags": {}}

    def serialize(self, groups=()):
        return {
            "id": "00000000-0000-4000-8000-000000000000",
            "revision": 0,
            "last_node_id": self._next_node - 1,
            "last_link_id": self._next_link - 1,
            "nodes": self.nodes,
            "links": self.links,
            "groups": list(groups),
            "config": {},
            "extra": {},
            "version": 0.4,
        }


TURBO_NOTE = """KREA 2 TURBO - SVDQuant W4A4

1. Checkpoint goes in ComfyUI/models/diffusion_models/.
   This graph expects an svdq checkpoint (it carries *.svdq_l1 /
   *.svdq_l2 tensors). The --format w4a4 / int8 / fp8 files have no
   low-rank branch and load with the stock UNETLoader instead.

2. cfg MUST be 1.0. Krea 2 Turbo is cfg-distilled, so the negative
   prompt is zeroed out (ConditioningZeroOut) rather than encoded.
   Raising cfg here degrades the image, it does not sharpen it.
   8 steps is what the checkpoint was distilled for.

3. Text encoder: any Qwen3-VL 4B in ComfyUI/models/text_encoders/,
   loaded with CLIPLoader type "krea2".
   VAE: qwen_image_vae.safetensors.

4. SLOW? Read the loader's "status" output. It names the kernel that
   will actually run. If it does not say cuda, ComfyUI has disabled
   comfy_kitchen's CUDA backend because torch was built against
   CUDA < 13 -- and the fallback dequantizes int4 in Python, so the
   checkpoint ends up slower than fp8. Install a cu130+ torch build.
   The Krea2 SVDQuant Env Check node answers this with no model loaded.

5. LoRA: use the Krea2 SVDQuant LoRA Loader, not the stock one. The
   stock loader cannot patch a quantized weight and skips the blocks."""

BASE_NOTE = """KREA 2 BASE (non-turbo) - SVDQuant W4A4

The loader node will be outlined in RED when you open this. That is
expected: the filename it wants does not exist until you build it in
step 1, and ComfyUI flags any dropdown value it cannot find. Build the
checkpoint, then reselect it in the loader.

1. There is no pre-built base checkpoint to download. Make one:

     python quantize_krea2.py raw.safetensors \\
         --format svdq --rank 64 --variant base

   That writes Krea2-Base-SVDQuant-W4A4-rank64.safetensors next to
   the source. ~54s without refinement, ~5.7min with it.
   The Krea2 SVDQuant Quantize node does the same thing from inside
   ComfyUI if you would rather not touch a terminal.

2. Unlike Turbo, this model is NOT cfg-distilled, so it uses real
   classifier-free guidance: cfg 3.5 and a real negative prompt
   (a second CLIPTextEncode, not ConditioningZeroOut).
   ~50 steps. These are starting points, tune them.

3. This is roughly 12x the sampling work of the 8-step Turbo graph.
   Every benchmark number in the README is Turbo at 8 steps - the
   ratios carry over, the absolute seconds do not.

4. Text encoder and VAE are the same as the Turbo graph.
   Same slow-generation checklist too: read the loader's status
   output first."""

PROMPT = ("A cluttered antique clockmaker's workshop seen through a cracked magnifying "
          "glass, brass gears laid out in a spiral on the workbench, a ginger cat asleep "
          "on a stack of leather-bound books, warm afternoon light with visible dust "
          "motes, photorealistic, 85mm lens, shallow depth of field")
NEGATIVE = "blurry, low resolution, jpeg artifacts, watermark, deformed, extra limbs"


def build_turbo():
    g = Graph()
    g.note(TURBO_NOTE, 0, 0, size=(400, 620), title="READ ME FIRST")

    loader = g.add("Krea2SVDQuantW4A4Loader", 1, 0,
                   outputs=[("MODEL", "MODEL"), ("STRING", "STRING")],
                   widgets=["Krea2-Turbo-SVDQuant-W4A4-rank64.safetensors"],
                   title="Krea2 SVDQuant W4A4 Loader", colour=TEAL, size=[400, 120])
    clip = g.add("CLIPLoader", 1, 1,
                 outputs=[("CLIP", "CLIP")],
                 widgets=["qwen3vl_4b_fp8_scaled.safetensors", "krea2", "default"],
                 title="Text encoder (Qwen3-VL 4B)", colour=TEAL, size=[400, 120])
    vae = g.add("VAELoader", 1, 2, outputs=[("VAE", "VAE")],
                widgets=["qwen_image_vae.safetensors"], title="VAE", colour=TEAL,
                size=[400, 80])

    pos = g.add("CLIPTextEncode", 2, 0, inputs=[("clip", "CLIP")],
                outputs=[("CONDITIONING", "CONDITIONING")], widgets=[PROMPT],
                title="Prompt", colour=GREEN, size=[400, 220])
    neg = g.add("ConditioningZeroOut", 2, 1.4, inputs=[("conditioning", "CONDITIONING")],
                outputs=[("CONDITIONING", "CONDITIONING")],
                title="Negative (zeroed - required at cfg 1.0)", colour=GREEN,
                size=[400, 60])
    latent = g.add("EmptySD3LatentImage", 2, 2.2, outputs=[("LATENT", "LATENT")],
                   widgets=[1024, 1024, 1], title="Latent 1024x1024", colour=GREEN,
                   size=[400, 120])

    sampler = g.add("KSampler", 3, 0,
                    inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                            ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                    outputs=[("LATENT", "LATENT")],
                    widgets=[987654321, "randomize", 8, 1.0, "euler", "simple", 1.0],
                    title="KSampler - 8 steps, cfg 1.0", colour=PURPLE, size=[400, 280])
    decode = g.add("VAEDecode", 4, 0, inputs=[("samples", "LATENT"), ("vae", "VAE")],
                   outputs=[("IMAGE", "IMAGE")], title="VAE Decode", colour=PURPLE,
                   size=[300, 60])
    save = g.add("SaveImage", 4, 0.7, inputs=[("images", "IMAGE")],
                 widgets=["krea2_turbo_svdq"], title="Save", colour=PURPLE, size=[400, 300])

    diag = g.add("Krea2SVDQuantDiagnostics", 3, 2.4, inputs=[("model", "MODEL")],
                 outputs=[("MODEL", "MODEL"), ("STRING", "STRING")],
                 widgets=["dispatch", 4096],
                 title="Diagnostics (optional - run if slow)", colour=YELLOW,
                 size=[400, 130])

    g.link(loader, "MODEL", sampler, "model")
    g.link(clip, "CLIP", pos, "clip")
    g.link(pos, "CONDITIONING", neg, "conditioning")
    g.link(pos, "CONDITIONING", sampler, "positive")
    g.link(neg, "CONDITIONING", sampler, "negative")
    g.link(latent, "LATENT", sampler, "latent_image")
    g.link(sampler, "LATENT", decode, "samples")
    g.link(vae, "VAE", decode, "vae")
    g.link(decode, "IMAGE", save, "images")
    g.link(loader, "MODEL", diag, "model")

    groups = [
        g.group("Load", (500, 20, 420, 620)),
        g.group("Prompt", (940, 20, 420, 620)),
        g.group("Sample", (1360, 20, 420, 640), colour="#8a4"),
    ]
    return g.serialize(groups)


def build_base():
    g = Graph()
    g.note(BASE_NOTE, 0, 0, size=(400, 560), title="READ ME FIRST")

    loader = g.add("Krea2SVDQuantW4A4Loader", 1, 0,
                   outputs=[("MODEL", "MODEL"), ("STRING", "STRING")],
                   widgets=["Krea2-Base-SVDQuant-W4A4-rank64.safetensors"],
                   title="Krea2 SVDQuant W4A4 Loader (base)", colour=TEAL, size=[400, 120])
    clip = g.add("CLIPLoader", 1, 1, outputs=[("CLIP", "CLIP")],
                 widgets=["qwen3vl_4b_fp8_scaled.safetensors", "krea2", "default"],
                 title="Text encoder (Qwen3-VL 4B)", colour=TEAL, size=[400, 120])
    vae = g.add("VAELoader", 1, 2, outputs=[("VAE", "VAE")],
                widgets=["qwen_image_vae.safetensors"], title="VAE", colour=TEAL,
                size=[400, 80])

    pos = g.add("CLIPTextEncode", 2, 0, inputs=[("clip", "CLIP")],
                outputs=[("CONDITIONING", "CONDITIONING")], widgets=[PROMPT],
                title="Positive prompt", colour=GREEN, size=[400, 200])
    neg = g.add("CLIPTextEncode", 2, 1.3, inputs=[("clip", "CLIP")],
                outputs=[("CONDITIONING", "CONDITIONING")], widgets=[NEGATIVE],
                title="Negative prompt (base uses real CFG)", colour=GREEN,
                size=[400, 140])
    latent = g.add("EmptySD3LatentImage", 2, 2.3, outputs=[("LATENT", "LATENT")],
                   widgets=[1024, 1024, 1], title="Latent 1024x1024", colour=GREEN,
                   size=[400, 120])

    sampler = g.add("KSampler", 3, 0,
                    inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                            ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                    outputs=[("LATENT", "LATENT")],
                    widgets=[987654321, "randomize", 50, 3.5, "euler", "simple", 1.0],
                    title="KSampler - 50 steps, cfg 3.5", colour=PURPLE, size=[400, 280])
    decode = g.add("VAEDecode", 4, 0, inputs=[("samples", "LATENT"), ("vae", "VAE")],
                   outputs=[("IMAGE", "IMAGE")], title="VAE Decode", colour=PURPLE,
                   size=[300, 60])
    save = g.add("SaveImage", 4, 0.7, inputs=[("images", "IMAGE")],
                 widgets=["krea2_base_svdq"], title="Save", colour=PURPLE, size=[400, 300])

    g.link(loader, "MODEL", sampler, "model")
    g.link(clip, "CLIP", pos, "clip")
    g.link(clip, "CLIP", neg, "clip")
    g.link(pos, "CONDITIONING", sampler, "positive")
    g.link(neg, "CONDITIONING", sampler, "negative")
    g.link(latent, "LATENT", sampler, "latent_image")
    g.link(sampler, "LATENT", decode, "samples")
    g.link(vae, "VAE", decode, "vae")
    g.link(decode, "IMAGE", save, "images")

    groups = [
        g.group("Load", (500, 20, 420, 560)),
        g.group("Prompt", (940, 20, 420, 620)),
        g.group("Sample", (1360, 20, 420, 640), colour="#8a4"),
    ]
    return g.serialize(groups)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, graph in (("krea2_turbo_svdquant_w4a4_t2i", build_turbo()),
                        ("krea2_base_svdquant_w4a4_t2i", build_base())):
        path = os.path.join(OUT_DIR, name + ".json")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(graph, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print("wrote {}  ({} nodes, {} links)".format(
            path, len(graph["nodes"]), len(graph["links"])))


if __name__ == "__main__":
    main()

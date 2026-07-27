"""Contact sheets for the fidelity_bench renders, so a human can judge them directly.

    python tools/contact_sheet.py --output-dir <ComfyUI/output> [--arm base] [--cell 640]

One sheet per prompt: columns are checkpoints, rows are seeds, every tile labelled. The
BF16 reference is forced into the first column so the eye has a fixed anchor to compare
against rather than hunting for it.

This exists because LPIPS answers "how far did it move", not "is it worse". Those come
apart: W4A4 shifts the sampling trajectory, so a checkpoint can score a large distance and
still look fine, or score a small one and mangle a line of text. The numbers rank, the
sheets adjudicate.

Deliberately no vision-model judge. The one used previously saturated -- 71 of 110 cells came
back a flat 10/10 -- so it produced a confident ranking with no discrimination behind it.

Needs Pillow, which ComfyUI already depends on.
"""

from __future__ import annotations

import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fidelity_bench as fb  # noqa: E402

LABEL_H = 28
PAD = 6
BG = (24, 24, 26)
FG = (232, 232, 236)
REF_FG = (150, 210, 255)


def _font(size: int = 16):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_sheet(cells: dict, checkpoints: list, seeds: list, cell_px: int, title: str):
    """`cells` is {(checkpoint, seed): path}. Missing entries render as an empty slot."""
    font = _font()
    title_h = LABEL_H + 8
    w = len(checkpoints) * (cell_px + PAD) + PAD
    h = title_h + len(seeds) * (cell_px + LABEL_H + PAD) + PAD
    sheet = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, PAD), title, fill=FG, font=_font(18))

    for col, ckpt in enumerate(checkpoints):
        for row, seed in enumerate(seeds):
            x = PAD + col * (cell_px + PAD)
            y = title_h + row * (cell_px + LABEL_H + PAD)
            label = "{}  seed {}".format(ckpt, seed)
            draw.text((x, y), label, font=font,
                      fill=REF_FG if ckpt == fb.REFERENCE else FG)
            path = cells.get((ckpt, seed))
            if path is None:
                draw.rectangle([x, y + LABEL_H, x + cell_px, y + LABEL_H + cell_px],
                               outline=(70, 70, 74))
                continue
            img = Image.open(path).convert("RGB")
            img.thumbnail((cell_px, cell_px), Image.LANCZOS)
            sheet.paste(img, (x, y + LABEL_H))
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True,
                    help="the ComfyUI output directory (reads its fb/ subfolder)")
    ap.add_argument("--arm", default=None, choices=["base", "lora"],
                    help="default: one set of sheets per arm present")
    ap.add_argument("--cell", type=int, default=640, help="tile size in px")
    ap.add_argument("--dest", default=None, help="default: <output-dir>/fb/sheets")
    args = ap.parse_args()

    out_root = os.path.join(args.output_dir, fb.SUBDIR)
    images = fb.index(out_root)
    if not images:
        raise SystemExit("no renders in {}".format(out_root))

    dest = args.dest or os.path.join(out_root, "sheets")
    os.makedirs(dest, exist_ok=True)

    arms = [args.arm] if args.arm else sorted({a for a, _ in images})
    written = 0
    for arm in arms:
        ckpts = sorted(c for a, c in images if a == arm)
        # Reference first, then whatever order the rest come in -- a moving anchor makes
        # side-by-side comparison much harder than it needs to be.
        if fb.REFERENCE in ckpts:
            ckpts = [fb.REFERENCE] + [c for c in ckpts if c != fb.REFERENCE]

        prompts, seeds = set(), set()
        for (a, _), cells in images.items():
            if a != arm:
                continue
            for pid, seed in cells:
                prompts.add(pid)
                seeds.add(seed)

        for pid in sorted(prompts):
            cells = {}
            for ckpt in ckpts:
                for (p, seed), path in images.get((arm, ckpt), {}).items():
                    if p == pid:
                        cells[(ckpt, seed)] = path
            sheet = build_sheet(cells, ckpts, sorted(seeds), args.cell,
                                "{}  [{}]".format(pid, arm))
            path = os.path.join(dest, "sheet_{}_{}.png".format(arm, pid))
            sheet.save(path)
            written += 1
            print("wrote {}".format(path))

    print("\n{} sheets in {}".format(written, dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())

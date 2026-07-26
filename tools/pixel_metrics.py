"""LPIPS / PSNR / SSIM for quantized checkpoints against a BF16 reference.

    python tools/pixel_metrics.py --dir <output-dir> [--out scores.csv]

Images are paired by filename: ``bench_<checkpoint>_<prompt_id>_00001_.png``, matched against
``bench_turbo_bf16_reference_<prompt_id>_00001_.png``.

Why this exists: the earlier sweep of these same images scored them with a vision model and
produced a ranking that cannot be defended. One seed per cell, and 71 of 110 cells came back a
flat 10/10 -- the judge had no discrimination left, so checkpoint means differed by 0.45 on a
10-point scale from single samples. LPIPS is deterministic and continuous, so it can separate
things a saturated judge cannot.

Read the output with two cautions:

* **A number is only meaningful against a noise floor.** Two BF16 runs at different seeds are
  not identical images either. Until that distance is measured, "LPIPS 0.27" does not tell you
  whether a checkpoint drifted or merely rolled different dice. Pass ``--noise-floor`` a pair
  of same-checkpoint / different-seed renders to print it alongside.
* **LPIPS measures divergence, not damage.** W4A4 shifts the sampling trajectory: the image
  lands somewhere else, which is not the same as landing somewhere worse. High LPIPS with
  intact aesthetics means drift. Do not read it as quality loss on its own.

Needs `lpips` (`pip install lpips`). SSIM is computed here rather than pulled from
scikit-image to keep the dependency list to one.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys

import torch
import torch.nn.functional as F

REFERENCE = "turbo_bf16_reference"
_NAME_RE = re.compile(r"^bench_(?P<ckpt>.+?)_(?P<prompt>\d{2}_[a-z_]+)_\d+_\.png$")


def _load(path: str, device) -> torch.Tensor:
    """PNG -> [1,3,H,W] in [-1,1], which is what LPIPS expects."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    t = torch.frombuffer(img.tobytes(), dtype=torch.uint8).clone()
    t = t.view(img.height, img.width, 3).permute(2, 0, 1).float().div_(255.0)
    return t.unsqueeze(0).mul_(2.0).sub_(1.0).to(device)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """On [0,1] data, so the peak is 1.0."""
    mse = F.mse_loss(a.add(1).div(2), b.add(1).div(2)).item()
    return float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Gaussian-windowed SSIM, averaged over channels. 11x11, sigma 1.5, the usual constants."""
    x, y = a.add(1).div(2), b.add(1).div(2)
    coords = torch.arange(11, dtype=torch.float32, device=x.device) - 5
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    window = (g[:, None] @ g[None, :]).expand(3, 1, 11, 11).contiguous()

    def blur(t):
        return F.conv2d(t, window, padding=5, groups=3)

    mu_x, mu_y = blur(x), blur(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = blur(x * x) - mu_x2
    sigma_y = blur(y * y) - mu_y2
    sigma_xy = blur(x * y) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2))
    return s.mean().item()


def index_images(directory: str) -> dict[str, dict[str, str]]:
    """{checkpoint: {prompt_id: path}} for every bench_*.png in `directory`."""
    found: dict[str, dict[str, str]] = {}
    for name in sorted(os.listdir(directory)):
        m = _NAME_RE.match(name)
        if m:
            found.setdefault(m["ckpt"], {})[m["prompt"]] = os.path.join(directory, name)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding the bench_*.png renders")
    ap.add_argument("--out", default=None, help="CSV to write (default: <dir>/pixel_metrics.csv)")
    ap.add_argument("--reference", default=REFERENCE, help="checkpoint name to treat as truth")
    ap.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"],
                    help="LPIPS backbone. alex matches what most published tables use")
    ap.add_argument("--noise-floor", nargs=2, metavar=("IMG_A", "IMG_B"),
                    help="two renders of the same checkpoint at different seeds; their "
                         "distance is the threshold any result below is indistinguishable from")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    try:
        import lpips
    except ImportError:
        raise SystemExit("pip install lpips")

    images = index_images(args.dir)
    if args.reference not in images:
        raise SystemExit("no reference renders named {!r} in {}; found: {}".format(
            args.reference, args.dir, ", ".join(sorted(images)) or "nothing"))
    refs = images[args.reference]
    device = torch.device(args.device)
    net = lpips.LPIPS(net=args.net).to(device).eval()

    floor = None
    if args.noise_floor:
        with torch.no_grad():
            a, b = (_load(p, device) for p in args.noise_floor)
            floor = {"lpips": net(a, b).item(), "psnr": psnr(a, b), "ssim": ssim(a, b)}
        print("noise floor (same checkpoint, different seed): LPIPS {lpips:.3f}  "
              "PSNR {psnr:.1f}  SSIM {ssim:.3f}\n".format(**floor))
    else:
        print("no --noise-floor given: the numbers below have no threshold to be compared "
              "against. Two BF16 runs at different seeds already differ.\n")

    rows = []
    per_ckpt: dict[str, list[float]] = {}
    with torch.no_grad():
        for ckpt in sorted(images):
            if ckpt == args.reference:
                continue
            for prompt_id, cand_path in sorted(images[ckpt].items()):
                ref_path = refs.get(prompt_id)
                if ref_path is None:
                    print("  skipping {} / {}: no reference render".format(ckpt, prompt_id))
                    continue
                a, b = _load(ref_path, device), _load(cand_path, device)
                if a.shape != b.shape:
                    print("  skipping {} / {}: {} vs {}".format(
                        ckpt, prompt_id, tuple(a.shape), tuple(b.shape)))
                    continue
                row = {"checkpoint": ckpt, "prompt_id": prompt_id,
                       "lpips": round(net(a, b).item(), 5),
                       "psnr": round(psnr(a, b), 3),
                       "ssim": round(ssim(a, b), 5)}
                rows.append(row)
                per_ckpt.setdefault(ckpt, []).append(row["lpips"])
                del a, b

    # sanity check: the reference against itself must be 0
    with torch.no_grad():
        first = next(iter(refs.values()))
        same = _load(first, device)
        self_lpips = net(same, same.clone()).item()
    print("sanity: reference vs itself LPIPS = {:.6f} (should be ~0)\n".format(self_lpips))

    out_path = args.out or os.path.join(args.dir, "pixel_metrics.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["checkpoint", "prompt_id", "lpips", "psnr", "ssim"])
        w.writeheader()
        w.writerows(rows)

    print("{:<32} {:>8} {:>8} {:>8} {:>8} {:>4}".format(
        "checkpoint", "LPIPS", "+-sd", "PSNR", "SSIM", "n"))
    summary = []
    for ckpt in sorted(per_ckpt, key=lambda c: statistics.mean(per_ckpt[c])):
        sub = [r for r in rows if r["checkpoint"] == ckpt]
        mean_l = statistics.mean(r["lpips"] for r in sub)
        sd = statistics.stdev([r["lpips"] for r in sub]) if len(sub) > 1 else 0.0
        mean_p = statistics.mean(r["psnr"] for r in sub if r["psnr"] != float("inf"))
        mean_s = statistics.mean(r["ssim"] for r in sub)
        flag = ""
        if floor and mean_l <= floor["lpips"]:
            flag = "  <- within seed noise"
        print("{:<32} {:>8.4f} {:>8.4f} {:>8.2f} {:>8.4f} {:>4}{}".format(
            ckpt, mean_l, sd, mean_p, mean_s, len(sub), flag))
        summary.append({"checkpoint": ckpt, "lpips": round(mean_l, 5), "lpips_sd": round(sd, 5),
                        "psnr": round(mean_p, 3), "ssim": round(mean_s, 5), "n": len(sub)})

    print("\nper prompt, mean LPIPS across checkpoints (higher = harder for W4A4):")
    for prompt_id in sorted({r["prompt_id"] for r in rows}):
        sub = [r["lpips"] for r in rows if r["prompt_id"] == prompt_id]
        print("  {:<24} {:.4f}".format(prompt_id, statistics.mean(sub)))

    json_path = os.path.splitext(out_path)[0] + "_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"reference": args.reference, "net": args.net, "noise_floor": floor,
                   "reference_self_lpips": self_lpips, "checkpoints": summary}, fh, indent=2)
    print("\nwrote {}\n      {}".format(out_path, json_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

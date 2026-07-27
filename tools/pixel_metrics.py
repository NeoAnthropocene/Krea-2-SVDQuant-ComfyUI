"""LPIPS / PSNR / SSIM implementations, shared by `fidelity_bench.py`.

Not a command-line tool. It used to be, scoring renders named
``bench_<checkpoint>_<prompt_id>_00001_.png`` -- a naming scheme nothing in this repo
produces any more, which made the CLI undriveable, and a methodology `fidelity_bench.py`
argues against in its own docstring (one seed per cell, marginal means). The measurement
code was worth keeping; the entry point was not.

Read any of these numbers with two cautions:

* **A number is only meaningful against a noise floor.** Two BF16 runs at different seeds are
  not identical images either. Until that distance is measured, "LPIPS 0.27" does not tell you
  whether a checkpoint drifted or merely rolled different dice. `fidelity_bench score` prints
  the floor per arm for exactly this reason.
* **LPIPS measures divergence, not damage.** W4A4 shifts the sampling trajectory: the image
  lands somewhere else, which is not the same as landing somewhere worse. High LPIPS with
  intact aesthetics means drift. Do not read it as quality loss on its own.

SSIM is computed here rather than pulled from scikit-image to keep the dependency list short.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def load_image(path: str, device) -> torch.Tensor:
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

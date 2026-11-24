"""Turn predicted maps into one grasp per image (proposal "Architecture", inference)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_kernel(sigma: float, device=None) -> torch.Tensor:
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (B, 1, H, W) maps with reflect padding."""
    if sigma <= 0:
        return x
    k = gaussian_kernel(sigma, x.device)
    r = len(k) // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


@torch.no_grad()
def maps_to_grasps(out: torch.Tensor, q_sigma: float = 2.0, ang_sigma: float = 2.0, w_sigma: float = 1.0,
                   w_scale: float = 150.0, jaw_ratio: float = 0.5) -> torch.Tensor:
    """(B, 4, H, W) = Q, cos2t, sin2t, W  ->  (B, 5) grasps (x, y, theta, w, h).

    Q is smoothed (as in GG-CNN, Morrison et al. 2018) and its arg-max is the
    grasp centre; cos/sin are smoothed separately so the angle average is valid.
    """
    B, _, H, Wd = out.shape
    q = blur(out[:, 0:1], q_sigma).flatten(1)
    idx = q.argmax(1)
    ys, xs = idx // Wd, idx % Wd
    b = torch.arange(B, device=out.device)
    c = blur(out[:, 1:2], ang_sigma)[b, 0, ys, xs]
    s = blur(out[:, 2:3], ang_sigma)[b, 0, ys, xs]
    w = blur(out[:, 3:4], w_sigma)[b, 0, ys, xs] * w_scale
    theta = 0.5 * torch.atan2(s, c)
    return torch.stack([xs.float(), ys.float(), theta, w, jaw_ratio * w], 1)

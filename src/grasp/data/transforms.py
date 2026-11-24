"""Training-time augmentation (proposal step 6) and the depth-noise model.

Geometric transforms are applied to the image *and* the grasp rectangles; label
maps are then re-drawn from the transformed rectangles, so the angle labels stay
exact under rotation.
"""
from __future__ import annotations

import cv2
import numpy as np

from .depth import inpaint_depth
from .labels import transform_rects


def add_depth_noise(depth: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean Gaussian depth noise (metres). Also used for the noise-robustness curve."""
    if sigma <= 0:
        return depth
    return (depth + rng.normal(0.0, sigma, depth.shape)).astype(np.float32)


def depth_dropout(depth: np.ndarray, rng: np.random.Generator, max_patches: int, size_px=(4, 16),
                  inpaint_radius: float = 3) -> np.ndarray:
    """Cut random rectangular holes and fill them like the real pipeline does (Telea inpainting)."""
    n = rng.integers(0, max_patches + 1)
    if n == 0:
        return depth
    H, W = depth.shape
    mask = np.zeros((H, W), np.uint8)
    for _ in range(n):
        h, w = rng.integers(size_px[0], size_px[1] + 1, size=2)
        y, x = rng.integers(0, H - h), rng.integers(0, W - w)
        mask[y:y + h, x:x + w] = 1
    return inpaint_depth(depth, mask.astype(bool), inpaint_radius)


def colour_jitter(rgb: np.ndarray, rng: np.random.Generator, brightness=0.2, contrast=0.2,
                  saturation=0.2) -> np.ndarray:
    x = rgb.astype(np.float32)
    x *= rng.uniform(1 - brightness, 1 + brightness)
    mean = x.mean()
    x = (x - mean) * rng.uniform(1 - contrast, 1 + contrast) + mean
    grey = x.mean(axis=2, keepdims=True)
    x = (x - grey) * rng.uniform(1 - saturation, 1 + saturation) + grey
    return np.clip(x, 0, 255).astype(np.uint8)


def random_affine(rng: np.random.Generator, size: int, rotate_deg: float, translate_px: float, zoom):
    """Rotation about the centre, magnification 1/zoom, then translation."""
    angle = rng.uniform(-rotate_deg, rotate_deg)
    scale = 1.0 / rng.uniform(zoom[0], zoom[1])
    M = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, scale)
    M[:, 2] += rng.uniform(-translate_px, translate_px, size=2)
    return M, np.deg2rad(angle), scale


def apply_affine(rgb, depth, rects, M, angle_rad, scale):
    H, W = depth.shape
    rgb = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    depth = cv2.warpAffine(depth, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rects = transform_rects(rects, M, angle_rad, scale)
    keep = (rects[:, 0] >= 0) & (rects[:, 0] < W) & (rects[:, 1] >= 0) & (rects[:, 1] < H)
    return rgb, depth, rects[keep]


class Augmenter:
    def __init__(self, cfg: dict, inpaint_radius: float = 3):
        self.cfg = cfg
        self.inpaint_radius = inpaint_radius

    def __call__(self, rgb, depth, rects, rng: np.random.Generator):
        c = self.cfg
        M, a, s = random_affine(rng, depth.shape[0], c["rotate_deg"], c["translate_px"], c["zoom"])
        rgb2, depth2, rects2 = apply_affine(rgb, depth, rects, M, a, s)
        if len(rects2) == 0:  # every grasp centre left the image: keep the sample unaugmented
            rgb2, depth2, rects2 = rgb, depth, np.asarray(rects, dtype=np.float64)
        rgb2 = colour_jitter(rgb2, rng, c["brightness"], c["contrast"], c["saturation"])
        depth2 = depth_dropout(depth2, rng, c["dropout_patches_max"], c["dropout_patch_px"], self.inpaint_radius)
        depth2 = add_depth_noise(depth2, c["depth_noise_sigma"], rng)
        return rgb2, depth2, rects2

"""Hand-coded PCA grasp heuristic (proposal "Baseline Model"). No learning; uses only depth.

Steps on the 224x224 height map (metres above the fitted table plane):
  1. table level = median of the image-border pixels;
  2. object = pixels more than ``height_thresh`` above it; keep the largest
     8-connected component (optionally ignoring components that touch the image
     border, e.g. the back wall, unless nothing else is left);
  3. grasp centre = mask centroid;
  4. PCA on the mask pixel coordinates -> long axis; close the gripper across
     it (opening axis = long axis + 90 deg);
  5. w = mask extent along the opening axis + ``margin_px``, capped at ``max_w_px``.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .data.labels import wrap_angle
from .metrics import make_pred

DEFAULTS = {"height_thresh": 0.01, "margin_px": 10, "max_w_px": 150, "exclude_border_components": True,
            "jaw_ratio": 0.5}


def object_mask(height: np.ndarray, height_thresh: float = 0.01, exclude_border_components: bool = True):
    border = np.concatenate([height[0], height[-1], height[1:-1, 0], height[1:-1, -1]])
    table = float(np.median(border))
    above = height > table + height_thresh
    lab, n = ndimage.label(above, structure=np.ones((3, 3)))
    if n == 0:
        return None, table
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    candidates = np.arange(1, n + 1)
    if exclude_border_components:
        edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        inner = np.array([c for c in candidates if c not in edge])
        if len(inner):
            sizes, candidates = sizes[inner - 1], inner
    return lab == candidates[int(np.argmax(sizes))], table


def pca_grasp(mask: np.ndarray, margin_px: float = 10, max_w_px: float = 150, jaw_ratio: float = 0.5):
    """Returns the grasp (x, y, theta, w, h) and the PCA long-axis angle."""
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    if len(xs) < 3:
        long_axis = 0.0
    else:
        cov = np.cov(np.stack([xs - cx, ys - cy]))
        evals, evecs = np.linalg.eigh(cov)
        vx, vy = evecs[:, np.argmax(evals)]
        long_axis = float(np.arctan2(-vy, vx))  # same convention as labels: CCW on screen
    theta = float(wrap_angle(long_axis + np.pi / 2))
    u = np.array([np.cos(theta), -np.sin(theta)])  # opening direction in image coords
    proj = (xs - cx) * u[0] + (ys - cy) * u[1]
    w = min(float(proj.max() - proj.min()) + margin_px, max_w_px)
    return make_pred(cx, cy, theta, w, jaw_ratio), long_axis


def predict(height: np.ndarray, cfg: dict | None = None) -> dict:
    """Baseline grasp for one height map. Falls back to the image centre if no object is found."""
    c = {**DEFAULTS, **(cfg or {})}
    mask, table = object_mask(height, c["height_thresh"], c["exclude_border_components"])
    if mask is None:
        H, W = height.shape
        return {"grasp": make_pred(W / 2, H / 2, 0.0, 40, c["jaw_ratio"]), "mask": np.zeros_like(height, bool),
                "table": table, "found": False}
    grasp, long_axis = pca_grasp(mask, c["margin_px"], c["max_w_px"], c["jaw_ratio"])
    return {"grasp": grasp, "mask": mask, "table": table, "long_axis": long_axis, "found": True}

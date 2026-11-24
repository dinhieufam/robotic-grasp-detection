"""Depth hole filling.

cv2.inpaint with Telea (2004) gives badly wrong values on float32 input in the
OpenCV builds we tested (4.14 and 5.0: a ramp comes back with errors >1 and
small negative values flip sign). The uint16 path is correct, so we map each
image's valid depth range onto uint16, inpaint, and map back.
"""
from __future__ import annotations

import cv2
import numpy as np

_U16 = 65535.0


def inpaint_depth(depth: np.ndarray, missing: np.ndarray, radius: float = 3) -> np.ndarray:
    """Fill ``missing`` pixels of a float depth map with Telea inpainting. Returns float32."""
    depth = np.asarray(depth, dtype=np.float32)
    missing = np.asarray(missing, dtype=bool)
    if not missing.any():
        return depth.copy()
    valid = depth[~missing]
    if valid.size == 0:
        return np.zeros_like(depth)
    lo, hi = float(valid.min()), float(valid.max())
    span = max(hi - lo, 1e-6)
    u = np.clip((depth - lo) / span, 0, 1) * _U16
    u[missing] = 0
    out = cv2.inpaint(np.round(u).astype(np.uint16), missing.astype(np.uint8), radius, cv2.INPAINT_TELEA)
    out = out.astype(np.float32) / _U16 * span + lo
    out[~missing] = depth[~missing]
    return out



def camera_centre(points: np.ndarray, pix: np.ndarray, n_sample: int = 3000, seed: int = 0) -> np.ndarray:
    """Camera centre in the point-cloud frame from a DLT fit of pixel <- 3D point.

    Cornell point clouds are not in the camera frame, but each point carries its
    pixel, so the 3x4 projection matrix P is recovered exactly (reprojection
    error ~0 px); the centre is the null space of P.
    """
    rng = np.random.default_rng(seed)
    s = rng.choice(len(points), size=min(n_sample, len(points)), replace=False)
    X, uv = points[s], pix[s].astype(np.float64)
    m, sc = X.mean(0), X.std()
    Xh = np.c_[(X - m) / sc, np.ones(len(X))]
    A = np.zeros((2 * len(X), 12))
    A[0::2, 0:4], A[0::2, 8:12] = Xh, -uv[:, [0]] * Xh
    A[1::2, 4:8], A[1::2, 8:12] = Xh, -uv[:, [1]] * Xh
    P = np.linalg.svd(A, full_matrices=False)[2][-1].reshape(3, 4)
    return -np.linalg.solve(P[:, :3], P[:, 3]) * sc + m


def fit_table_plane(points: np.ndarray, toward: np.ndarray, thresh: float = 0.005, iters: int = 300,
                    n_sample: int = 6000, seed: int = 0):
    """RANSAC plane fit (Fischler & Bolles, 1981) + least-squares refit on the inliers.

    ``points`` (N, 3) in metres. Returns unit normal ``n`` and offset ``d`` with
    the plane n.x + d = 0, oriented so that ``toward`` (the camera) is on the
    positive side; n.x + d is then the height above the table.
    """
    rng = np.random.default_rng(seed)
    pts = points[rng.choice(len(points), size=min(n_sample, len(points)), replace=False)]
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        inl = np.abs(pts @ n - n @ a) < thresh
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    q = pts[best]
    centroid = q.mean(0)
    n = np.linalg.svd(q - centroid, full_matrices=False)[2][-1]
    d = -n @ centroid
    if n @ toward + d < 0:
        n, d = -n, -d
    return n, d, best_n / len(pts)

"""Cornell grasping dataset: raw readers and per-image preprocessing.

Raw layout (Kaggle mirror ``oneoneliu/cornell-grasp``)::

    raw/01..10/pcdXXXX.txt      point cloud (PCD v.7 ascii: x y z rgb index), mm
    raw/01..10/pcdXXXXr.png     640x480 RGB
    raw/01..10/pcdXXXXcpos.txt  positive rectangles, 4 lines "x y" per rectangle
    z.txt (processedData.zip)   "<image id> <object id> <name> <background>" per rectangle
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .depth import camera_centre, fit_table_plane, inpaint_depth
from .labels import corners_to_rect, is_degenerate

H_RAW, W_RAW = 480, 640


def list_images(raw_dir: str) -> dict[int, str]:
    """image id -> path prefix (".../pcdXXXX"), excluding background images."""
    out = {}
    for f in glob.glob(os.path.join(raw_dir, "[0-9][0-9]", "pcd[0-9][0-9][0-9][0-9]r.png")):
        img_id = int(re.search(r"pcd(\d{4})r\.png$", f).group(1))
        out[img_id] = f[: -len("r.png")]
    return dict(sorted(out.items()))


def load_object_ids(z_path: str) -> dict[int, tuple[int, str]]:
    """image id -> (object id, object name) from the original z.txt."""
    out = {}
    with open(z_path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                out[int(p[0])] = (int(p[1]), p[2])
    return out


def load_points_from_pcd(pcd_path: str):
    """Returns (points (N, 3) in metres, pixel (N, 2) as (col, row)). Point-cloud frame, not camera frame."""
    pts = np.loadtxt(pcd_path, skiprows=10, dtype=np.float64)
    idx = pts[:, 4].astype(np.int64)
    return pts[:, :3] / 1000.0, np.stack([idx % W_RAW, idx // W_RAW], axis=1)


def load_depth_from_pcd(pcd_path: str) -> np.ndarray:
    """480x640 'depth' = Euclidean norm of each point (metres, 0 = missing), as in GG-CNN
    (Morrison et al., 2018). Note: the norm is taken in the point-cloud frame."""
    xyz, pix = load_points_from_pcd(pcd_path)
    depth = np.zeros((H_RAW, W_RAW), dtype=np.float32)
    depth[pix[:, 1], pix[:, 0]] = np.linalg.norm(xyz, axis=1)
    return depth


def height_above_table(xyz, pix, crop_box, dcfg, seed=0):
    """480x640 map of signed height above the RANSAC table plane (metres, NaN = missing).
    The plane is fitted to the points inside ``crop_box`` = (x0, y0, size)."""
    x0, y0, S = crop_box
    inside = (pix[:, 0] >= x0) & (pix[:, 0] < x0 + S) & (pix[:, 1] >= y0) & (pix[:, 1] < y0 + S)
    t = dcfg.get("table_fit", {})
    cam = camera_centre(xyz, pix, seed=seed)
    n, d, inlier_frac = fit_table_plane(xyz[inside], cam, t.get("thresh_m", 0.005), t.get("iters", 300), seed=seed)
    h = np.full((H_RAW, W_RAW), np.nan, dtype=np.float32)
    h[pix[:, 1], pix[:, 0]] = xyz @ n + d
    return h, inlier_frac


def load_corners(cpos_path: str) -> list[np.ndarray]:
    """List of (4, 2) corner arrays (x, y). NaN rows are kept (handled later)."""
    vals = []
    with open(cpos_path) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                vals.append([float(p[0]), float(p[1])])
    vals = np.array(vals, dtype=np.float64).reshape(-1, 2)
    n = len(vals) // 4
    return [vals[4 * i: 4 * i + 4] for i in range(n)]


@dataclass
class Processed:
    img_id: int
    status: str                      # "ok" or the reason it was dropped
    rgb: np.ndarray | None = None    # (224, 224, 3) uint8
    depth: np.ndarray | None = None  # (224, 224) float32 metres: height above table (or mean-subtracted depth), clipped
    holes: np.ndarray | None = None  # (224, 224) bool, depth was missing before inpainting
    rects: np.ndarray | None = None  # (N, 5) in 224-px coordinates
    crop_xy: tuple[int, int] = (0, 0)  # top-left of the crop in the raw image
    missing_frac: float = 0.0
    counts: dict = field(default_factory=dict)  # per-rectangle drop counts
    table_inliers: float = float("nan")         # fraction of crop points on the fitted table plane


def preprocess_image(prefix: str, img_id: int, cfg: dict) -> Processed:
    """Proposal steps 1-3 for one Cornell image."""
    min_size = cfg["labels"]["min_rect_size_px"]
    S, OUT = cfg["crop"]["size_px"], cfg["crop"]["out_px"]
    counts = {"rects_raw": 0, "rects_degenerate": 0, "rects_outside": 0}

    # Step 1: labels -> (x, y, theta, w, h); drop degenerate.
    corners = load_corners(prefix + "cpos.txt")
    counts["rects_raw"] = len(corners)
    good = [c for c in corners if not is_degenerate(c, min_size)]
    counts["rects_degenerate"] = len(corners) - len(good)
    if not good:
        return Processed(img_id, "no_valid_rects", counts=counts)

    # Step 3a: crop window centred on the mean grasp centre, kept inside the image.
    centre = np.mean([c.mean(axis=0) for c in good], axis=0)
    x0 = int(np.clip(round(centre[0] - S / 2), 0, W_RAW - S))
    y0 = int(np.clip(round(centre[1] - S / 2), 0, H_RAW - S))

    inside = [np.all((c[:, 0] >= x0) & (c[:, 0] < x0 + S) & (c[:, 1] >= y0) & (c[:, 1] < y0 + S)) for c in good]
    counts["rects_outside"] = int(len(good) - sum(inside))
    if counts["rects_outside"]:
        if cfg["crop"]["outside_policy"] == "drop_image":
            return Processed(img_id, "rect_outside_crop", counts=counts)
        good = [c for c, k in zip(good, inside) if k]

    # Step 2: depth from the point cloud; missing check on the crop.
    xyz, pix = load_points_from_pcd(prefix + ".txt")
    dcfg = cfg["depth"]
    if dcfg["normalise"] == "table":
        # Height above the table plane (see PLAN.md § Deviations).
        depth, table_inliers = height_above_table(xyz, pix, (x0, y0, S), dcfg, seed=img_id)
        missing = np.isnan(depth)
        depth = np.nan_to_num(depth)
    elif dcfg["normalise"] == "mean":
        depth = np.zeros((H_RAW, W_RAW), dtype=np.float32)
        depth[pix[:, 1], pix[:, 0]] = np.linalg.norm(xyz, axis=1)
        missing, table_inliers = depth <= 0, float("nan")
    else:
        raise ValueError(dcfg["normalise"])
    m_crop = missing[y0:y0 + S, x0:x0 + S]
    missing_frac = float(m_crop.mean())
    if missing_frac > cfg["crop"]["max_missing_depth"]:
        return Processed(img_id, "missing_depth", missing_frac=missing_frac, counts=counts)

    # Inpaint (with a small margin of real context so edges are well filled).
    pad = 10
    ya, yb = max(0, y0 - pad), min(H_RAW, y0 + S + pad)
    xa, xb = max(0, x0 - pad), min(W_RAW, x0 + S + pad)
    filled = inpaint_depth(depth[ya:yb, xa:xb], missing[ya:yb, xa:xb], dcfg["inpaint_radius_px"])
    d_crop = filled[y0 - ya:y0 - ya + S, x0 - xa:x0 - xa + S]

    # Step 3b: resize to 224 and rescale rectangles.
    rgb = cv2.cvtColor(cv2.imread(prefix + "r.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb[y0:y0 + S, x0:x0 + S], (OUT, OUT), interpolation=cv2.INTER_AREA)
    d_out = cv2.resize(d_crop, (OUT, OUT), interpolation=cv2.INTER_AREA)
    holes = cv2.resize(m_crop.astype(np.uint8), (OUT, OUT), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Step 2 (end): proposal subtracts the image mean; height maps are already table-relative. Clip.
    if dcfg["normalise"] == "mean":
        d_out = d_out - d_out.mean()
    d_out = np.clip(d_out, -dcfg["clip"], dcfg["clip"]).astype(np.float32)

    k = OUT / S
    rects = np.stack([corners_to_rect(c) for c in good])
    rects[:, 0] = (rects[:, 0] - x0) * k
    rects[:, 1] = (rects[:, 1] - y0) * k
    rects[:, 3:5] *= k

    return Processed(img_id, "ok", rgb=rgb, depth=d_out, holes=holes, rects=rects.astype(np.float32),
                     crop_xy=(x0, y0), missing_frac=missing_frac, counts=counts, table_inliers=table_inliers)

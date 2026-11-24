"""Shared evaluation: score one predicted grasp per image with the rectangle metric."""
from __future__ import annotations

import csv
import time

import numpy as np

from .metrics import match_grasp


def evaluate(predict_fn, data, img_ids, angle_thresh_deg=30.0, iou_thresh=0.25):
    """``predict_fn(row_index) -> (x, y, theta, w, h)``. Returns (summary, per-image rows)."""
    rows, times = [], []
    for i in img_ids:
        k = data.index[int(i)]
        t0 = time.perf_counter()
        g = np.asarray(predict_fn(k), dtype=np.float64)
        times.append(time.perf_counter() - t0)
        m = match_grasp(g, data.rects(k), angle_thresh_deg, iou_thresh)
        rows.append({"img_id": int(i), "obj_id": int(data.obj_ids[k]), "obj_name": str(data.obj_names[k]),
                     "x": g[0], "y": g[1], "theta": g[2], "w": g[3], "h": g[4], **m})
    correct = np.array([r["correct"] for r in rows])
    lo, hi = wilson_ci(int(correct.sum()), len(rows))
    summary = {"n": len(rows), "accuracy": float(correct.mean()), "ci95": [lo, hi],
               "failures_angle": int(sum((not r["correct"]) and r["min_angle_diff_deg"] > angle_thresh_deg
                                         for r in rows)),
               "failures_iou_only": int(sum((not r["correct"]) and r["min_angle_diff_deg"] <= angle_thresh_deg
                                            for r in rows)),
               "ms_per_image_mean": 1000 * float(np.mean(times)), "ms_per_image_median": 1000 * float(np.median(times))}
    return summary, rows


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion (small test sets!)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

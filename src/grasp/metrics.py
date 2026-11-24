"""Rectangle metric (Jiang et al., 2011) used for the baseline and every model.

A predicted grasp is correct if, for at least one labelled rectangle,
  * the angle difference (mod 180 deg) is <= 30 deg, and
  * the IoU (Jaccard index) of the two rectangles is > 25%.

Grasps are (x, y, theta, w, h) as defined in grasp.data.labels. A prediction
only has an opening ``w``; its jaw size is ``h = jaw_ratio * w`` (GG-CNN
convention, Morrison et al. 2018, jaw_ratio = 0.5).
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from .data.labels import rect_to_corners

ANGLE_THRESH_DEG = 30.0
IOU_THRESH = 0.25


def angle_diff(t1, t2):
    """Absolute difference of grasp angles, treating theta and theta + pi as equal. In [0, pi/2]."""
    d = np.mod(np.asarray(t1) - np.asarray(t2), np.pi)
    return np.minimum(d, np.pi - d)


def rect_polygon(rect) -> Polygon:
    return Polygon(rect_to_corners(rect))


def rect_iou(r1, r2) -> float:
    p1, p2 = rect_polygon(r1), rect_polygon(r2)
    inter = p1.intersection(p2).area
    union = p1.area + p2.area - inter
    return float(inter / union) if union > 0 else 0.0


def match_grasp(pred, gts, angle_thresh_deg: float = ANGLE_THRESH_DEG, iou_thresh: float = IOU_THRESH) -> dict:
    """Score one predicted rectangle against all labelled rectangles of an image.

    Returns ``correct`` plus diagnostics: the best IoU among angle-matching
    labels, the best IoU overall and the smallest angle difference (degrees).
    """
    gts = np.asarray(gts, dtype=np.float64).reshape(-1, 5)
    ad = np.rad2deg(angle_diff(pred[2], gts[:, 2]))
    ious = np.array([rect_iou(pred, g) for g in gts])
    ok_angle = ad <= angle_thresh_deg
    correct = bool(np.any(ok_angle & (ious > iou_thresh)))
    return {"correct": correct,
            "best_iou_angle_ok": float(ious[ok_angle].max()) if ok_angle.any() else 0.0,
            "best_iou": float(ious.max()), "min_angle_diff_deg": float(ad.min())}


def make_pred(x, y, theta, w, jaw_ratio: float = 0.5) -> np.ndarray:
    return np.array([x, y, theta, w, jaw_ratio * w], dtype=np.float64)

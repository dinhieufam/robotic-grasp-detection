"""Grasp rectangle representation and pixel-wise label maps.

Convention (used everywhere in this repo):
    A grasp is ``(x, y, theta, w, h)`` in pixel units.
    * ``x`` = column, ``y`` = row of the rectangle centre (image coords, y down).
    * ``theta`` = angle of the gripper *opening* axis, measured counter-clockwise
      on screen from the +x axis (i.e. ``atan2(-dy, dx)``), wrapped to
      ``(-pi/2, pi/2]`` because a parallel-jaw grasp is symmetric under 180 deg.
    * ``w`` = gripper opening (length along the opening axis).
    * ``h`` = jaw size (length perpendicular to the opening axis).

The direction of the opening axis in image coordinates is ``(cos t, -sin t)``.
"""
from __future__ import annotations

import numpy as np
from skimage.draw import polygon as _fill_polygon

W_SCALE = 150.0  # W label = w / W_SCALE (proposal step 4)


def wrap_angle(theta):
    """Wrap angle(s) to (-pi/2, pi/2]."""
    theta = np.asarray(theta, dtype=np.float64)
    return np.pi / 2 - np.mod(np.pi / 2 - theta, np.pi)


def corners_to_rect(corners: np.ndarray) -> np.ndarray:
    """Cornell 4-corner rectangle -> (x, y, theta, w, h).

    ``corners`` is (4, 2) as (x, y) = (col, row), in the order given by the
    dataset: edge p0->p1 is along the gripper opening, p1->p2 along the jaw.
    """
    c = np.asarray(corners, dtype=np.float64)
    centre = c.mean(axis=0)
    dx, dy = c[1] - c[0]
    theta = wrap_angle(np.arctan2(-dy, dx))
    w = np.hypot(dx, dy)
    h = np.hypot(*(c[2] - c[1]))
    return np.array([centre[0], centre[1], theta, w, h])


def rect_to_corners(rect) -> np.ndarray:
    """(x, y, theta, w, h) -> (4, 2) corners as (x, y), same order as Cornell."""
    x, y, t, w, h = rect
    u = np.array([np.cos(t), -np.sin(t)]) * w / 2  # opening axis
    v = np.array([np.sin(t), np.cos(t)]) * h / 2   # jaw axis (perpendicular)
    c = np.array([x, y])
    return np.stack([c - u - v, c + u - v, c + u + v, c - u + v])


def is_degenerate(corners: np.ndarray, min_size: float = 1.0) -> bool:
    """NaN/inf corners or a (near) zero-width/height rectangle."""
    if not np.all(np.isfinite(corners)):
        return True
    _, _, _, w, h = corners_to_rect(corners)
    return bool(w < min_size or h < min_size)


def transform_rects(rects: np.ndarray, M: np.ndarray, angle_rad: float, scale: float) -> np.ndarray:
    """Apply a similarity transform (as built by cv2.getRotationMatrix2D + shift).

    ``M`` is the 2x3 matrix applied to image coordinates; ``angle_rad`` is the
    counter-clockwise on-screen rotation it contains and ``scale`` its zoom.
    """
    rects = np.asarray(rects, dtype=np.float64).reshape(-1, 5).copy()
    if len(rects) == 0:
        return rects
    xy1 = np.concatenate([rects[:, :2], np.ones((len(rects), 1))], axis=1)
    rects[:, :2] = xy1 @ M.T
    rects[:, 2] = wrap_angle(rects[:, 2] + angle_rad)
    rects[:, 3:5] *= scale
    return rects


def draw_label_maps(rects: np.ndarray, shape=(224, 224), shrink: float = 1 / 3,
                    w_scale: float = W_SCALE) -> np.ndarray:
    """Pixel-wise targets (proposal step 4).

    Returns float32 array (4, H, W): Q, cos2theta, sin2theta, W.
    Each rectangle's middle third (along the opening axis, full jaw size) gets
    Q = 1, its cos/sin(2 theta) and W = w / w_scale (clipped to [0, 1]).
    Pixels outside every rectangle have all-zero targets. Later rectangles
    overwrite earlier ones where they overlap.
    """
    out = np.zeros((4, *shape), dtype=np.float32)
    for x, y, t, w, h in np.asarray(rects).reshape(-1, 5):
        corners = rect_to_corners((x, y, t, w * shrink, h))
        rr, cc = _fill_polygon(corners[:, 1], corners[:, 0], shape=shape)
        out[0, rr, cc] = 1.0
        out[1, rr, cc] = np.cos(2 * t)
        out[2, rr, cc] = np.sin(2 * t)
        out[3, rr, cc] = np.clip(w / w_scale, 0.0, 1.0)
    return out

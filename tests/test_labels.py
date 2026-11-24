import cv2
import numpy as np
import pytest

from grasp.data.labels import (corners_to_rect, draw_label_maps, is_degenerate, rect_to_corners,
                               transform_rects, wrap_angle)
from grasp.data.splits import image_wise_split, object_wise_split
from grasp.data.transforms import apply_affine, random_affine


def test_wrap_angle_range():
    t = np.linspace(-10, 10, 2001)
    w = wrap_angle(t)
    assert np.all(w > -np.pi / 2 - 1e-12) and np.all(w <= np.pi / 2 + 1e-12)
    assert np.allclose(np.cos(2 * w), np.cos(2 * t)) and np.allclose(np.sin(2 * w), np.sin(2 * t))
    assert wrap_angle(-np.pi / 2) == pytest.approx(np.pi / 2)
    assert wrap_angle(np.pi / 2) == pytest.approx(np.pi / 2)


def test_corners_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        r = np.array([rng.uniform(0, 200), rng.uniform(0, 200), rng.uniform(-1.5, 1.5),
                      rng.uniform(5, 80), rng.uniform(5, 40)])
        assert np.allclose(corners_to_rect(rect_to_corners(r)), r)


def test_angle_convention_screen_ccw():
    # Opening edge pointing right and *up* on screen (row decreases) -> +45 deg.
    c = np.array([[0, 10], [10, 0], [12, 2], [2, 12]], float)
    assert corners_to_rect(c)[2] == pytest.approx(np.pi / 4)
    # Horizontal opening edge -> 0; vertical -> pi/2 regardless of direction.
    assert corners_to_rect(np.array([[0, 0], [10, 0], [10, 5], [0, 5]], float))[2] == pytest.approx(0)
    assert corners_to_rect(np.array([[0, 0], [0, 10], [5, 10], [5, 0]], float))[2] == pytest.approx(np.pi / 2)


def test_degenerate():
    assert is_degenerate(np.array([[0, 0], [np.nan, 0], [1, 1], [0, 1]]))
    assert is_degenerate(np.array([[0, 0], [0, 0], [0, 5], [0, 5]], float))
    assert not is_degenerate(np.array([[0, 0], [10, 0], [10, 5], [0, 5]], float))


def test_label_map_middle_third():
    rect = np.array([100, 50, 0.0, 60, 20])  # horizontal opening, 60 px wide, 20 px jaw
    y = draw_label_maps(rect[None], (224, 224))
    q = y[0]
    rows, cols = np.nonzero(q)
    assert cols.min() >= 100 - 10 - 1 and cols.max() <= 100 + 10 + 1     # 60/3 = 20 px along opening
    assert rows.min() >= 50 - 10 - 1 and rows.max() <= 50 + 10 + 1       # full jaw size
    assert np.allclose(y[1][q > 0], 1) and np.allclose(y[2][q > 0], 0)    # cos 0, sin 0
    assert np.allclose(y[3][q > 0], 60 / 150)
    assert y[:, q == 0].sum() == 0


def test_label_map_angle_values():
    t = 0.6
    y = draw_label_maps(np.array([[112, 112, t, 60, 30]]), (224, 224))
    m = y[0] > 0
    assert np.allclose(y[1][m], np.cos(2 * t)) and np.allclose(y[2][m], np.sin(2 * t))


def test_affine_matches_corner_transform():
    """Transforming (x, y, theta, w, h) must equal transforming its corners with the same matrix."""
    rng = np.random.default_rng(1)
    rects = np.array([[100, 120, 0.3, 50, 20], [60, 80, -1.2, 30, 15]], float)
    for _ in range(20):
        M, a, s = random_affine(rng, 224, 180, 20, (0.8, 1.0))
        out = transform_rects(rects, M, a, s)
        for r, o in zip(rects, out):
            c = rect_to_corners(r)
            c2 = np.concatenate([c, np.ones((4, 1))], 1) @ M.T
            assert np.allclose(corners_to_rect(c2), o, atol=1e-6) or \
                np.allclose(rect_to_corners(o)[[2, 3, 0, 1]], c2, atol=1e-6)


def test_affine_moves_image_with_labels():
    """A bright blob under the grasp centre stays under the transformed grasp centre."""
    rng = np.random.default_rng(2)
    rgb = np.zeros((224, 224, 3), np.uint8)
    depth = np.zeros((224, 224), np.float32)
    cv2.circle(depth, (150, 90), 4, 1.0, -1)
    rects = np.array([[150, 90, 0.2, 40, 20]], float)
    for _ in range(10):
        M, a, s = random_affine(rng, 224, 180, 20, (0.8, 1.0))
        _, d2, r2 = apply_affine(rgb, depth, rects, M, a, s)
        if len(r2):
            x, y = np.round(r2[0, :2]).astype(int)
            assert d2[y, x] > 0.5


def test_splits_disjoint_objects():
    img = np.arange(100)
    obj = img // 4
    s = object_wise_split(img, obj, seed=0)
    objs = {k: set(obj[np.isin(img, v)]) for k, v in s.items()}
    assert not (objs["train"] & objs["val"]) and not (objs["train"] & objs["test"]) and not (objs["val"] & objs["test"])
    assert sum(len(v) for v in s.values()) == 100
    s2 = image_wise_split(img, seed=0)
    assert sorted(sum(s2.values(), [])) == list(range(100))


def test_inpaint_depth_ramp_with_negatives():
    from grasp.data.depth import inpaint_depth
    yy = np.linspace(-0.3, 0.3, 64, dtype=np.float32)[:, None] * np.ones((1, 64), np.float32)
    m = np.zeros_like(yy, bool)
    m[20:26, 30:36] = True
    m[40, 10] = True
    out = inpaint_depth(yy, m, 3)
    assert np.abs(out - yy)[m].max() < 0.03          # float32 cv2.inpaint gives > 1 here
    assert np.array_equal(out[~m], yy[~m])
    assert out.min() >= yy.min() - 1e-6 and out.max() <= yy.max() + 1e-6



def _synthetic_scene(seed=0):
    """Tilted table (z=0 plane in a rotated frame), a 4 cm box and a wall; pinhole camera."""
    rng = np.random.default_rng(seed)
    pts = [np.c_[rng.uniform(-.3, .3, (6000, 2)), np.zeros(6000)]]                       # table
    pts.append(np.c_[rng.uniform(-.05, .05, (800, 2)), np.full(800, 0.04)])               # box top
    pts.append(np.c_[rng.uniform(-.3, .3, 1500), np.full(1500, 0.3), rng.uniform(0, .3, 1500)])  # wall
    X = np.concatenate(pts)
    a = 0.4
    Rw = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Xw = X @ Rw.T + np.array([0.5, -0.2, 0.3])          # arbitrary world frame
    C = np.array([0.5, -0.2, 0.3]) + Rw @ np.array([0.05, -0.4, 0.7])   # above the table, oblique view
    # camera looking at the table centre
    f = (np.array([0.5, -0.2, 0.3]) - C); f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    Xc = (Xw - C) @ np.stack([r, -u, f]).T
    pix = np.c_[500 * Xc[:, 0] / Xc[:, 2] + 320, 500 * Xc[:, 1] / Xc[:, 2] + 240]
    return Xw, pix, C, Rw @ np.array([0, 0, 1.0]), X[:, 2]


def test_camera_centre_dlt():
    from grasp.data.depth import camera_centre
    Xw, pix, C, _, _ = _synthetic_scene()
    assert np.allclose(camera_centre(Xw, pix), C, atol=1e-6)


def test_table_plane_height_ignores_box_and_wall():
    from grasp.data.depth import fit_table_plane
    Xw, pix, C, normal, height = _synthetic_scene()
    n, d, inl = fit_table_plane(Xw, C, thresh=0.003)
    assert abs(abs(n @ normal) - 1) < 1e-6 and n @ C + d > 0
    assert np.allclose(Xw @ n + d, height, atol=1e-4)     # signed height above the table
    assert 0.6 < inl < 0.8

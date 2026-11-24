import numpy as np
import pytest

from grasp.metrics import angle_diff, make_pred, match_grasp, rect_iou


def test_angle_diff_wraps():
    d = np.deg2rad
    assert np.rad2deg(angle_diff(d(85), d(-85))) == pytest.approx(10)     # across the +-90 seam
    assert np.rad2deg(angle_diff(d(10), d(190))) == pytest.approx(0)       # 180 deg symmetric
    assert np.rad2deg(angle_diff(d(0), d(90))) == pytest.approx(90)
    assert np.rad2deg(angle_diff(d(-30), d(30))) == pytest.approx(60)


def test_iou_known_values():
    a = np.array([50, 50, 0, 20, 10.0])
    assert rect_iou(a, a) == pytest.approx(1)
    assert rect_iou(a, np.array([60, 50, 0, 20, 10.0])) == pytest.approx(100 / 300)   # half overlap
    assert rect_iou(a, np.array([200, 50, 0, 20, 10.0])) == 0
    # 90 deg rotated copy of a 20x10 rect: overlap is a 10x10 square -> 100 / 300
    assert rect_iou(a, np.array([50, 50, np.pi / 2, 20, 10.0])) == pytest.approx(100 / 300)
    # theta and theta + pi describe the same rectangle
    b = np.array([50, 50, 0.4, 30, 12.0])
    assert rect_iou(b, b + np.array([0, 0, np.pi, 0, 0])) == pytest.approx(1)


def test_match_thresholds():
    gt = np.array([[100, 100, 0.0, 40, 20]])
    assert match_grasp(make_pred(100, 100, 0.0, 40), gt)["correct"]
    assert match_grasp(make_pred(100, 100, np.deg2rad(29), 40), gt)["correct"]
    assert not match_grasp(make_pred(100, 100, np.deg2rad(31), 40), gt)["correct"]       # angle fails
    assert match_grasp(make_pred(100, 100, np.deg2rad(179), 40), gt)["correct"]          # wrap-around
    r = match_grasp(make_pred(150, 100, 0.0, 40), gt)                                    # no overlap
    assert not r["correct"] and r["best_iou"] == 0 and r["min_angle_diff_deg"] == 0


def test_match_any_label():
    gts = np.array([[100, 100, 0.0, 40, 20], [50, 50, np.pi / 2, 30, 15]])
    assert match_grasp(make_pred(50, 50, np.pi / 2 - 0.1, 30), gts)["correct"]
    # right place of label 1 but angle of label 2 -> wrong (both conditions must hold for the same label)
    assert not match_grasp(make_pred(100, 100, np.pi / 2, 40), gts)["correct"]

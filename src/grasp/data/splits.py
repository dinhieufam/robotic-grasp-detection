"""Train/val/test splits (proposal step 5)."""
from __future__ import annotations

import numpy as np


def _cut(n: int, fractions) -> tuple[int, int]:
    a = int(round(fractions[0] * n))
    b = a + int(round(fractions[1] * n))
    return a, b


def object_wise_split(img_ids, obj_ids, fractions=(0.7, 0.15, 0.15), seed=0) -> dict[str, list[int]]:
    """Split *objects* 70/15/15; every image of an object goes to the same split."""
    img_ids, obj_ids = np.asarray(img_ids), np.asarray(obj_ids)
    objs = np.unique(obj_ids)
    np.random.default_rng(seed).shuffle(objs)
    a, b = _cut(len(objs), fractions)
    groups = {"train": set(objs[:a]), "val": set(objs[a:b]), "test": set(objs[b:])}
    return {k: sorted(int(i) for i, o in zip(img_ids, obj_ids) if o in g) for k, g in groups.items()}


def image_wise_split(img_ids, fractions=(0.7, 0.15, 0.15), seed=0) -> dict[str, list[int]]:
    """Split images at random (same object may appear in train and test)."""
    ids = np.array(sorted(img_ids))
    np.random.default_rng(seed).shuffle(ids)
    a, b = _cut(len(ids), fractions)
    return {"train": sorted(map(int, ids[:a])), "val": sorted(map(int, ids[a:b])), "test": sorted(map(int, ids[b:]))}

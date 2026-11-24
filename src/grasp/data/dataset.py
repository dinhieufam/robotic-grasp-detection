"""PyTorch Dataset over the preprocessed .npz cache."""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import draw_label_maps
from .transforms import Augmenter

CHANNELS = {"rgbd": (0, 1, 2, 3), "rgb": (0, 1, 2), "d": (3,)}


class CachedGraspData:
    """Loads the whole cache once (≈400 MB for Cornell) and indexes it by image id."""

    def __init__(self, npz_path: str):
        z = np.load(npz_path)
        self.rgb, self.depth, self.holes = z["rgb"], z["depth"], z["holes"]
        self.rects_flat, self.offsets = z["rects"], z["rect_offsets"]
        self.img_ids, self.obj_ids = z["img_ids"], z["obj_ids"]
        self.obj_names = z["obj_names"]
        self.index = {int(i): k for k, i in enumerate(self.img_ids)}

    def rects(self, k: int) -> np.ndarray:
        return self.rects_flat[self.offsets[k]:self.offsets[k + 1]]

    def __len__(self):
        return len(self.img_ids)


class GraspDataset(Dataset):
    """Returns ``x`` (C, H, W), ``y`` (4, H, W) = Q, cos2t, sin2t, W, and the rectangles.

    RGB is scaled to [-0.5, 0.5]; depth is already mean-subtracted metres.
    ``augment`` is the config's ``augment`` dict (training) or None.
    """

    def __init__(self, data: CachedGraspData, split_json: str, subset: str, channels: str = "rgbd",
                 augment: dict | None = None, label_cfg: dict | None = None, seed: int = 0):
        with open(split_json) as f:
            ids = json.load(f)[subset]
        self.data = data
        self.rows = [data.index[i] for i in ids]
        self.channels = CHANNELS[channels]
        self.augment = Augmenter(augment) if augment else None
        self.label_cfg = label_cfg or {"shrink": 1 / 3, "w_scale": 150.0}
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        k = self.rows[i]
        rgb, depth, rects = self.data.rgb[k], self.data.depth[k], self.data.rects(k)
        if self.augment is not None:
            rng = np.random.default_rng([self.seed, self.epoch, k])
            rgb, depth, rects = self.augment(rgb, depth, rects, rng)
        x = np.concatenate([rgb.astype(np.float32).transpose(2, 0, 1) / 255.0 - 0.5, depth[None]], axis=0)
        y = draw_label_maps(rects, depth.shape, self.label_cfg["shrink"], self.label_cfg["w_scale"])
        return {"x": torch.from_numpy(np.ascontiguousarray(x[list(self.channels)])), "y": torch.from_numpy(y),
                "rects": torch.as_tensor(np.asarray(rects, dtype=np.float32)), "img_id": int(self.data.img_ids[k])}


def collate(batch):
    return {"x": torch.stack([b["x"] for b in batch]), "y": torch.stack([b["y"] for b in batch]),
            "rects": [b["rects"] for b in batch], "img_id": torch.tensor([b["img_id"] for b in batch])}

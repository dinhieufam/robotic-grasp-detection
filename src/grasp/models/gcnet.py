"""Our fully convolutional grasp network (proposal "Architecture").

    input (C, 224, 224)
    enc1  Conv 5x5 s2 -> 32 @112   \\
    enc2  Conv 3x3 s2 -> 64 @56     |  Conv-BN-ReLU
    enc3  Conv 3x3 s2 -> 128 @28   /
    5 x residual block (two 3x3 Conv-BN, identity skip) @28        (He et al., 2016)
    dec1  ConvT 4x4 s2 -> 64 @56,  concat enc2 -> 128               (U-Net skips, Ronneberger et al., 2015)
    dec2  ConvT 4x4 s2 -> 32 @112, concat enc1 -> 64
    dec3  ConvT 4x4 s2 -> 32 @224
    heads 1x1 Conv: Q (sigmoid), cos2t (tanh), sin2t (tanh), W (sigmoid)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn_relu(cin, cout, k, s):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def deconv_bn_relu(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                                  nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class GraspNet(nn.Module):
    def __init__(self, in_channels: int = 4, widths=(32, 64, 128), n_res: int = 5, dropout: float = 0.0):
        super().__init__()
        c1, c2, c3 = widths
        self.enc1 = conv_bn_relu(in_channels, c1, 5, 2)
        self.enc2 = conv_bn_relu(c1, c2, 3, 2)
        self.enc3 = conv_bn_relu(c2, c3, 3, 2)
        self.res = nn.Sequential(*[ResidualBlock(c3) for _ in range(n_res)])
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.dec1 = deconv_bn_relu(c3, c2)
        self.dec2 = deconv_bn_relu(2 * c2, c1)
        self.dec3 = deconv_bn_relu(2 * c1, c1)
        self.head_q = nn.Conv2d(c1, 1, 1)
        self.head_cos = nn.Conv2d(c1, 1, 1)
        self.head_sin = nn.Conv2d(c1, 1, 1)
        self.head_w = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        z = self.drop(self.res(self.enc3(e2)))
        d = self.dec1(z)
        d = self.dec2(torch.cat([d, e2], 1))
        d = self.dec3(torch.cat([d, e1], 1))
        return torch.cat([torch.sigmoid(self.head_q(d)), torch.tanh(self.head_cos(d)),
                          torch.tanh(self.head_sin(d)), torch.sigmoid(self.head_w(d))], 1)


def build_model(cfg: dict, in_channels: int) -> GraspNet:
    return GraspNet(in_channels, tuple(cfg.get("widths", (32, 64, 128))), cfg.get("n_res", 5), cfg.get("dropout", 0.0))

"""Custom attention modules: SimAM and EMA for Ultralytics YOLO models.

Citations:
- SimAM: [E1] L. Yang, R.-Y. Zhang, L. Li, X. Xie, "SimAM: A Simple,
  Parameter-Free Attention Module for Convolutional Neural Networks,"
  ICML 2021.
- EMA: [E2] D. Ouyang et al., "Efficient Multi-Scale Attention Module with
  Cross-Spatial Learning," ICASSP 2023.
- Ultralytics dispatch: ultralytics==8.4.177 nn/tasks.py parse_model line 103
  (globals()[m]). Because SimAM and EMA are custom modules not in base_modules,
  parse_model falls back to `else: c2 = ch[f]` and passes yaml args directly:
  SimAM(*args), EMA(*args). The channel count `c` must therefore be declared
  explicitly in YAML, identical to CBAM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from cbam import CBAM, CBAMStd


class SimAM(nn.Module):
    """Simple, Parameter-Free Attention Module for ConvNets (SimAM).

    Reference: L. Yang et al., ICML 2021.
    Given feature map X in R^{B x C x H x W}:
        d = (x - mean)^2
        y = d / (4 * (sum(d)/n + lambda)) + 0.5
        out = x * sigmoid(y)
    Computed in fp32 internally and cast back to input dtype to prevent
    AMP overflow.
    """

    def __init__(self, c: int | None = None, e_lambda: float = 1e-4) -> None:
        super().__init__()
        self.c = c
        self.e_lambda = e_lambda
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w - 1
        orig_dtype = x.dtype
        x_f = x.float()

        d = (x_f - x_f.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        out = x_f * self.act(e_inv)
        return out.to(orig_dtype)


class EMA(nn.Module):
    """Efficient Multi-Scale Attention Module with Cross-Spatial Learning.

    Reference: D. Ouyang et al., "Efficient Multi-Scale Attention Module with
    Cross-Spatial Learning," ICASSP 2023 (arXiv:2305.13563).
    Official repo: https://github.com/YOLOonMe/EMA-attention-module
    """

    def __init__(self, c: int, factor: int = 8) -> None:
        super().__init__()
        assert factor > 0, f"factor must be positive, got {factor}"
        assert c % factor == 0 and c // factor > 0, (
            f"c ({c}) must be divisible by factor ({factor})"
        )
        self.groups = factor
        c_sub = c // factor

        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.gn = nn.GroupNorm(c_sub, c_sub)
        self.conv1x1 = nn.Conv2d(
            c_sub, c_sub, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.conv3x3 = nn.Conv2d(
            c_sub, c_sub, kernel_size=3, stride=1, padding=1, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        c_sub = c // self.groups
        group_x = x.reshape(b * self.groups, c_sub, h, w)

        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(
            group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        )
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(
            self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )
        x12 = x2.reshape(b * self.groups, c_sub, -1)
        x21 = self.softmax(
            self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )
        x22 = x1.reshape(b * self.groups, c_sub, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * self.groups, 1, h, w
        )
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


ATTN_TYPES = (CBAM, CBAMStd, SimAM, EMA)


def register_all() -> None:
    """Register SimAM and EMA into ultralytics.nn.tasks for parse_model.

    Cited: ultralytics==8.4.177 nn/tasks.py lines 103, 2162-2167.
    """
    import ultralytics.nn.tasks as T

    T.SimAM = SimAM
    T.EMA = EMA

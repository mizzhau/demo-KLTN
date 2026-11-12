"""CBAM (Convolutional Block Attention Module) implementation.

Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
Protocol: Protocol_MaTran_ThucNghiem_v2.md (sections 5.3, C.3)
and Protocol_v3_addendum.md (A.2).
API line citation: ultralytics==8.4.177 nn/tasks.py lines 2162-2167
(module dispatch via globals() in parse_model).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CBAM(nn.Module):
    """CBAM module: channel attention first, spatial attention second.

    Supports two initialization modes:
    - init="open": Open-gate initialization v2 (Protocol v2.1 section 5.3).
      Channel gate ~ 0.982, Spatial gate ~ 0.88.
    - init="default": PyTorch default layer initialization (used for B3 model).
    """

    def __init__(
        self,
        c: int,
        r: int = 16,
        k: int = 7,
        init: str = "open",
    ) -> None:
        super().__init__()
        self.init_mode = init
        self._capture: bool = False
        self._mc: torch.Tensor | None = None
        self._ms: torch.Tensor | None = None
        h = max(c // r, 1)
        self.fc1 = nn.Linear(c, h)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(h, c)
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

        if init == "open":
            # Open-gate init v2 (Protocol 5.3):
            # Channel gate ~ 0.982; Spatial gate ~ 0.88
            nn.init.zeros_(self.fc2.weight)
            nn.init.constant_(self.fc2.bias, 2.0)
            nn.init.constant_(self.bn.weight, 1.0)
            nn.init.constant_(self.bn.bias, 2.0)
        elif init == "default":
            pass
        else:
            raise ValueError(
                f"Unknown init mode: {init}, expected 'open' or 'default'"
            )

    def mlp(self, v: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(v)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention: AvgPool + MaxPool over spatial dims (2, 3)
        mc = torch.sigmoid(
            self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3)))
        )[:, :, None, None]
        xc = x * mc
        # Spatial attention: AvgPool + MaxPool across channel dim 1
        ms = torch.sigmoid(
            self.bn(
                self.conv(
                    torch.cat(
                        [xc.mean(1, keepdim=True), xc.amax(1, keepdim=True)],
                        dim=1,
                    )
                )
            )
        )
        if self._capture:
            self._mc = mc.detach()
            self._ms = ms.detach()
        return xc * ms


class CBAMStd(CBAM):
    """CBAM with default PyTorch initialization (alias for B3 yaml)."""

    def __init__(self, c: int, r: int = 16, k: int = 7) -> None:
        super().__init__(c, r, k, init="default")


def register() -> None:
    """Register CBAM and Exp2 attention modules in ultralytics.nn.tasks.

    Cited: ultralytics==8.4.177 nn/tasks.py lines 2162-2167.
    """
    import ultralytics.nn.tasks as T
    from attention import register_all

    T.CBAM = CBAM
    T.CBAMStd = CBAMStd
    register_all()

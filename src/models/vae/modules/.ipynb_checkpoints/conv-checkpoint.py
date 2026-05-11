"""3D convolution with symmetric replicate padding."""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import cast_tuple


class CausalConv3d(nn.Module):
    """3D conv with symmetric replicate padding on T, H, W (kernel // 2 each side).

    Class name kept as CausalConv3d for drop-in compatibility with the original
    causal variant; padding is actually symmetric on all dims.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode: str = "replicate",
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        stride = cast_tuple(stride, 3)
        dilation = cast_tuple(dilation, 3)
        t_k, h_k, w_k = kernel_size
        self.pad_mode = pad_mode

        # F.pad order: (W_left, W_right, H_top, H_bottom, T_front, T_back).
        self.causal_padding = (w_k // 2, w_k // 2, h_k // 2, h_k // 2, t_k // 2, t_k // 2)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.causal_padding, mode=self.pad_mode)
        return self.conv(x)
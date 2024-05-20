# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_

from .utils import BaseModule


class Conv2dKan(ABC, nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()

        self.__w = nn.Parameter(
            th.ones(in_channels, out_channels, kernel_size * kernel_size, 1)
        )
        self.__c = nn.Parameter(
            th.ones(in_channels, out_channels, kernel_size * kernel_size, 1, n)
        )

        normal_(self.__w)
        normal_(self.__c, 0, 1e-3)

        self.__in_channels = in_channels
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding

    @abstractmethod
    def _act(self, x: th.Tensor) -> th.Tensor:
        pass

    @abstractmethod
    def _residual_act(self, x: th.Tensor) -> th.Tensor:
        pass

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 4
        assert x.size(1) == self.__in_channels

        b, c, h, w = x.size()

        output_height = (
            h - self.__kernel_size + 2 * self.__padding
        ) // self.__stride + 1
        output_width = (
            w - self.__kernel_size + 2 * self.__padding
        ) // self.__stride + 1

        windowed_x = (
            F.unfold(
                x,
                self.__kernel_size,
                1,
                self.__padding,
                self.__stride,
            )
            .view(b, c, self.__kernel_size**2, -1)
            .unsqueeze(2)
        )

        return th.sum(
            th.sum(
                self.__w
                * (
                    self._residual_act(windowed_x)
                    + th.sum(self.__c * self._act(windowed_x), dim=-1)
                ),
                dim=1,  # sum over input space
            ),
            dim=2,  # sum over window
        ).view(b, -1, output_height, output_width)


class Conv2dKanLayers(ABC, nn.Sequential, BaseModule):
    def __init__(
        self,
        channels: List[Tuple[int, int]],
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__(
            *[
                self._get_conv_kan(c_i, c_o, kernel_size, stride, padding)
                for c_i, c_o in channels
            ]
            + [nn.Flatten(1, -1)]
        )

    @abstractmethod
    def _get_conv_kan(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> Conv2dKan:
        pass

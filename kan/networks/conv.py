# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch as th
from torch import nn
from torch.nn import functional as F

from .hermite import hermite
from .utils import BaseModule


class AbstractConv2dKan(ABC, nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()

        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding
        self.__out_channels = out_channels

        self.__kan = self._get_kan(in_channels, out_channels)

    @abstractmethod
    def _get_kan(self, input_space: int, output_space: int) -> nn.Module:
        pass

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, h, w = x.size()

        output_height = (
            h - self.__kernel_size + 2 * self.__padding
        ) // self.__stride + 1
        output_width = (
            w - self.__kernel_size + 2 * self.__padding
        ) // self.__stride + 1

        out: th.Tensor = (
            self.__kan(
                F.unfold(
                    x,
                    self.__kernel_size,
                    1,
                    self.__padding,
                    self.__stride,
                ).view(b, c, 1, self.__kernel_size**2, -1)
            )
            .view(b, self.__out_channels, self.__kernel_size**2, -1)
            .sum(dim=2)
            .view(b, -1, output_height, output_width)
        )
        return out


class AbstractConv2dKanLayers(ABC, nn.Sequential, BaseModule):
    def __init__(
        self,
        channels: List[Tuple[int, int]],
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
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
    ) -> AbstractConv2dKan:
        pass


# Hermite
class _HermiteConv2d(nn.Module):
    def __init__(
        self, input_space: int, output_space: int, n: int, kernel_size: int
    ) -> None:
        super().__init__()

        self.__n = n
        self.__w = nn.Parameter(
            th.randn(input_space, output_space, kernel_size * kernel_size, 1)
        )
        self.__c = nn.Parameter(
            1e-3
            * th.randn(
                input_space,
                output_space,
                kernel_size * kernel_size,
                1,
                self.__n,
            )
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.sum(
            self.__w
            * (F.mish(x) + th.sum(self.__c * hermite(x, self.__n), dim=-1)),
            dim=1,
        )


class HermiteConv2dKan(AbstractConv2dKan):
    def __init__(
        self,
        input_space: int,
        output_space: int,
        n: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        self.__n = n
        self.__kernel_size = kernel_size
        super().__init__(
            input_space, output_space, kernel_size, stride, padding
        )

    def _get_kan(self, input_space: int, output_space: int) -> nn.Module:
        return _HermiteConv2d(
            input_space, output_space, self.__n, self.__kernel_size
        )


class HermiteConv2dKanLayers(AbstractConv2dKanLayers):
    def __init__(
        self,
        layers: List[Tuple[int, int]],
        n: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        self.__n = n
        super().__init__(layers, kernel_size, stride, padding)

    def _get_conv_kan(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> AbstractConv2dKan:
        return HermiteConv2dKan(
            in_channels, out_channels, self.__n, kernel_size, stride, padding
        )

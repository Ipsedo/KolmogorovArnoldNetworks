# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch as th
from torch.nn import functional as F

from .conv import Conv2dKan, Conv2dKanLayers
from .linear import LinearKAN, LinearKanLayers


def hermite(x: th.Tensor, n: int) -> th.Tensor:
    h_s = [th.ones(*x.size(), device=x.device), x]

    for i in range(1, n):
        h_s.append(x * h_s[i] - i * h_s[i - 1])

    return th.slice_copy(th.stack(h_s, dim=-1), -1, 1) / th.exp(
        th.lgamma(th.arange(2, n + 2, device=x.device)) / 2
    )


class HermiteKAN(LinearKAN):

    def __init__(self, input_space: int, output_space: int, n: int) -> None:
        super().__init__(input_space, output_space, n)
        self.__n = n

    def _act(self, x: th.Tensor) -> th.Tensor:
        return hermite(x, self.__n)

    def _residual_act(self, x: th.Tensor) -> th.Tensor:
        return F.mish(x)


class HermiteKanLayers(LinearKanLayers):
    def __init__(self, layers: List[Tuple[int, int]], n: int) -> None:
        self.__n = n
        super().__init__(layers)

    def _get_layer(self, input_space: int, output_space: int) -> LinearKAN:
        return HermiteKAN(input_space, output_space, self.__n)


# Conv
class HermiteConv2dKan(Conv2dKan):
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
        super().__init__(
            input_space, output_space, self.__n, kernel_size, stride, padding
        )

    def _act(self, x: th.Tensor) -> th.Tensor:
        return hermite(x, self.__n)

    def _residual_act(self, x: th.Tensor) -> th.Tensor:
        return F.mish(x)


class HermiteConv2dKanLayers(Conv2dKanLayers):
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
    ) -> Conv2dKan:
        return HermiteConv2dKan(
            in_channels, out_channels, self.__n, kernel_size, stride, padding
        )

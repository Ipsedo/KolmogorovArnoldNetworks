# -*- coding: utf-8 -*-
from typing import Callable, List, Tuple

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_, xavier_normal_

from .utils import ActivationFunction, BaseModule


# pylint: disable=too-many-instance-attributes
class Conv2dKan(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        act_fun: ActivationFunction,
        res_act_fun: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        super().__init__()

        self.__act_fun = act_fun
        self.__res_act_fun = res_act_fun

        self.__w = nn.Parameter(
            th.ones(in_channels, out_channels, kernel_size * kernel_size, 1)
        )

        self.__c = nn.Parameter(
            th.ones(
                in_channels,
                out_channels,
                kernel_size * kernel_size,
                1,
                self.__act_fun.get_size(),
            )
        )

        xavier_normal_(self.__w)
        normal_(self.__c, 0, 1e-1)

        self.__in_channels = in_channels
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding

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

        windowed_x = F.unfold(
            x,
            self.__kernel_size,
            1,
            self.__padding,
            self.__stride,
        ).view(b, c, 1, self.__kernel_size**2, -1)

        return th.sum(
            th.sum(
                self.__w
                * (
                    self.__res_act_fun(windowed_x)
                    + th.sum(self.__c * self.__act_fun(windowed_x), dim=-1)
                ),
                dim=1,  # sum over input space
            ),
            dim=2,  # sum over window
        ).view(b, -1, output_height, output_width)


class Conv2dKanLayers(nn.Sequential, BaseModule):
    def __init__(
        self,
        channels: List[Tuple[int, int]],
        kernel_sizes: List[int],
        strides: List[int],
        paddings: List[int],
        act_fun: ActivationFunction,
        res_act_fun: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        super().__init__(
            *[
                Conv2dKan(c_i, c_o, k, s, p, act_fun, res_act_fun)
                for (c_i, c_o), k, s, p in zip(
                    channels, kernel_sizes, strides, paddings
                )
            ]
            + [nn.Flatten(1, -1)]
        )

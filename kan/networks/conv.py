# -*- coding: utf-8 -*-
from typing import Callable

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_, xavier_normal_

from .utils import ActivationFunction


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

        self.__w_b = nn.Parameter(
            th.ones(in_channels, out_channels, kernel_size * kernel_size, 1)
        )

        self.__w_s = nn.Parameter(
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

        xavier_normal_(self.__w_b, 1)
        normal_(self.__w_s, 0, 1e-3)
        normal_(self.__c, 0, 1e-1)

        self.__in_channels = in_channels
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding

    def __get_output_size(self, size: int) -> int:
        return (
            size - self.__kernel_size + 2 * self.__padding
        ) // self.__stride + 1

    def __unfold(self, x: th.Tensor) -> th.Tensor:
        b, c, _, _ = x.size()
        return F.unfold(
            x,
            self.__kernel_size,
            1,
            self.__padding,
            self.__stride,
        ).view(b, c, 1, self.__kernel_size**2, -1)

    def __activation(self, windowed_x: th.Tensor) -> th.Tensor:
        # sum over function approximation
        return th.sum(
            self.__w_b * self.__res_act_fun(windowed_x)
            + self.__w_s
            * th.sum(self.__c * self.__act_fun(windowed_x), dim=-1),
            dim=1,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 4
        assert x.size(1) == self.__in_channels

        b, _, h, w = x.size()

        output_height = self.__get_output_size(h)
        output_width = self.__get_output_size(w)

        # sum over window : dim=2
        return th.sum(
            self.__activation(self.__unfold(x)),
            dim=2,
        ).view(b, -1, output_height, output_width)

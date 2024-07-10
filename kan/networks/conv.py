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

        self._w_b = nn.Parameter(
            th.ones(out_channels, in_channels * kernel_size * kernel_size, 1)
        )

        self._w_s = nn.Parameter(
            th.ones(out_channels, in_channels * kernel_size * kernel_size, 1)
        )

        self._c = nn.Parameter(
            th.ones(
                self.__act_fun.get_size(),
                out_channels,
                in_channels * kernel_size * kernel_size,
                1,
            )
        )

        xavier_normal_(self._w_b, 1.0 / kernel_size**2)
        # xavier_normal_(self._w_s, 1e-3)
        normal_(self._c, 0, 1e-1 / kernel_size**2)

        self._in_channels = in_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

    def __get_output_size(self, size: int) -> int:
        return (
            size - self._kernel_size + 2 * self._padding
        ) // self._stride + 1

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 4
        assert x.size(1) == self._in_channels

        _, _, h, w = x.size()

        output_height = self.__get_output_size(h)
        output_width = self.__get_output_size(w)

        out_unfolded = F.unfold(
            x,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
        )

        out: th.Tensor = th.sum(
            self.__res_act_fun(out_unfolded).unsqueeze(1) * self._w_b
            + self._w_s
            * th.sum(
                self.__act_fun(out_unfolded).unsqueeze(2) * self._c,
                dim=1,
            ),
            dim=2,
        ).unflatten(-1, (output_height, output_width))

        return out

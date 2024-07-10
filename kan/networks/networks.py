# -*- coding: utf-8 -*-
from typing import Callable, List, Tuple

import torch as th
from torch import nn

from .conv import Conv2dKan
from .linear import LinearKAN
from .utils import ActivationFunction, InfoModule


class LinearKanLayers(nn.Sequential, InfoModule):
    def __init__(
        self,
        layers: List[Tuple[int, int]],
        act_fun: ActivationFunction,
        res_act_fun: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        super().__init__(
            *[LinearKAN(c_i, c_o, act_fun, res_act_fun) for c_i, c_o in layers]
        )


class Conv2dKanLayers(nn.Sequential, InfoModule):
    def __init__(
        self,
        channels: List[Tuple[int, int]],
        kernel_sizes: List[int],
        strides: List[int],
        paddings: List[int],
        linear_sizes: List[Tuple[int, int]],
        act_fun: ActivationFunction,
        res_act_fun: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        assert (
            len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        )

        conv_layers = [
            Conv2dKan(c_i, c_o, k, s, p, act_fun, res_act_fun)
            for (c_i, c_o), k, s, p in zip(
                channels, kernel_sizes, strides, paddings
            )
        ]

        flatten_layer = [nn.Flatten(1, -1)]

        clf_layers = [
            LinearKAN(i, o, act_fun, res_act_fun) for i, o in linear_sizes
        ]

        super().__init__(*conv_layers + flatten_layer + clf_layers)

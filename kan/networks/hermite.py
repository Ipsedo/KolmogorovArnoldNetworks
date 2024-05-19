# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch as th
from torch import nn
from torch.nn import functional as F

from .abstract_kan import AbstractKAN, AbstractKanLayers


def hermite(x: th.Tensor, n: int) -> th.Tensor:  # sans H_0
    raise NotImplementedError("broken hermite")

    # h = th.ones(x.size(-1), n + 1, device=x.device)
    #
    # h[:, :, :, 1] = x
    # for i in range(1, n):
    #     h[:, :, :, i + 1] = x * h[:, :, :, i] - i * h[:, :, :, i - 1]
    #
    # h = h[:, :, :, 1:] / th.exp(
    #     th.lgamma(th.arange(2, n + 2, device=x.device)[None, None, None, :])
    #     / 2
    # )
    #
    # return h


class HermiteKAN(AbstractKAN):

    def __init__(self, input_space: int, output_space: int, n: int) -> None:
        super().__init__(input_space, output_space)

        self.__n = n
        self.__c = nn.Parameter(th.randn(input_space, output_space, n))

    def _activation_function(self, x: th.Tensor) -> th.Tensor:
        return th.sum(self.__c * hermite(x, self.__n), dim=-1)

    def _residual_activation_function(self, x: th.Tensor) -> th.Tensor:
        return F.mish(x)


class HermiteKanLayers(AbstractKanLayers):
    def __init__(self, layers: List[Tuple[int, int]], n: int) -> None:
        self.__n = n
        super().__init__(layers)

    def _get_layer(self, input_space: int, output_space: int) -> AbstractKAN:
        return HermiteKAN(input_space, output_space, self.__n)

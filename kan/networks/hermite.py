# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch as th
from torch import nn
from torch.nn import functional as F

from .abstract_kan import AbstractKAN, AbstractKanLayers


def hermite(x: th.Tensor, n: int) -> th.Tensor:
    h_s = [th.ones(*x.size(), device=x.device), x]

    for i in range(1, n):
        h_s.append(x * h_s[i] - i * h_s[i - 1])

    return th.slice_copy(th.stack(h_s, dim=-1), -1, 1) / th.exp(
        th.lgamma(th.arange(2, n + 2, device=x.device)) / 2
    )


class HermiteKAN(AbstractKAN):

    def __init__(self, input_space: int, output_space: int, n: int) -> None:
        super().__init__(input_space, output_space)

        self.__n = n
        self.__c = nn.Parameter(1e-1 * th.randn(input_space, output_space, n))

    def _activation_function(self, x: th.Tensor) -> th.Tensor:
        return th.sum(
            self.__c * hermite(x, self.__n),
            dim=-1,
        )

    def _residual_activation_function(self, x: th.Tensor) -> th.Tensor:
        return F.mish(x)


class HermiteKanLayers(AbstractKanLayers):
    def __init__(self, layers: List[Tuple[int, int]], n: int) -> None:
        self.__n = n
        super().__init__(layers)

    def _get_layer(self, input_space: int, output_space: int) -> AbstractKAN:
        return HermiteKAN(input_space, output_space, self.__n)

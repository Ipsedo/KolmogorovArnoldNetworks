# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch as th
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.init import xavier_normal_

from .abstract_kan import AbstractKAN, AbstractKanLayers


def b_spline(
    x: th.Tensor, k: int, n: int, x_min: float = 0.0, x_max: float = 1.0
) -> th.Tensor:
    assert len(x.size()) == 3
    x = x.unsqueeze(-1)

    def __knots(_i: th.Tensor) -> th.Tensor:
        return _i / n * (x_max - x_min) + x_min

    i_s = th.arange(-k, n, device=x.device)[None, None, None, :]

    def __b_spline(curr_i_s: th.Tensor, curr_k: int) -> th.Tensor:
        if curr_k == 0:
            return th.logical_and(
                th.le(__knots(curr_i_s), x), th.lt(x, __knots(curr_i_s + 1))
            ).to(th.float)

        return __b_spline(curr_i_s, curr_k - 1) * (x - __knots(curr_i_s)) / (
            __knots(curr_i_s + curr_k) - __knots(curr_i_s)
        ) + __b_spline(curr_i_s + 1, curr_k - 1) * (
            __knots(curr_i_s + curr_k + 1) - x
        ) / (
            __knots(curr_i_s + curr_k + 1) - __knots(curr_i_s + 1)
        )

    return __b_spline(i_s, k)


class SplineKAN(AbstractKAN):
    def __init__(
        self, input_space: int, output_space: int, degree: int, n: int
    ) -> None:
        super().__init__(input_space, output_space)

        self.__k = degree
        self.__n = n
        self.__c = Parameter(
            th.randn(1, output_space, input_space, self.__n + self.__k)
        )

        xavier_normal_(self.__c, gain=1e-8)

    def _activation_function(self, x: Tensor) -> Tensor:
        return th.sum(self.__c * b_spline(x, self.__k, self.__n), dim=-1)

    def _residual_activation_function(self, x: Tensor) -> Tensor:
        return F.mish(x)


class SplineKanLayers(AbstractKanLayers):
    def __init__(
        self, layers: List[Tuple[int, int]], degree: int, grid_size: int
    ) -> None:
        self.__degree = degree
        self.__grid_size = grid_size
        super().__init__(layers)

    def _get_layer(self, input_space: int, output_space: int) -> AbstractKAN:
        return SplineKAN(
            input_space, output_space, self.__degree, self.__grid_size
        )

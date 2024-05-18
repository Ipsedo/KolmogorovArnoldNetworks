# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch as th
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F

from .abstract_kan import AbstractKAN, AbstractKanLayers


def b_spline(x: th.Tensor, k: int) -> th.Tensor:
    _, n = x.size()[:2]

    offset = k // 2

    def __knots(_i: th.Tensor) -> th.Tensor:
        return (_i + offset) / (n + offset * 2)

    i_s = th.arange(n, device=x.device).unsqueeze(0)

    def __b_spline(curr_i_s: th.Tensor, curr_k: int) -> th.Tensor:
        if curr_k == 0:
            return th.logical_and(
                th.ge(x, __knots(curr_i_s)), th.lt(x, __knots(curr_i_s + 1))
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
        self, input_space: int, output_space: int, degree: int
    ) -> None:
        super().__init__(input_space, output_space)

        self.__k = degree

        self.__c = Parameter(th.zeros(1, output_space, input_space))

    def _activation_function(self, x: Tensor) -> Tensor:
        return F.mish(x)

    def _residual_activation_function(self, x: Tensor) -> Tensor:
        b_i_s = b_spline(x, self.__k)
        return self.__c * b_i_s


class SplineKanLayers(AbstractKanLayers):
    def __init__(self, layers: List[Tuple[int, int]], degree: int) -> None:
        self.__degree = degree
        super().__init__(layers)

    def _get_layer(self, input_space: int, output_space: int) -> AbstractKAN:
        return SplineKAN(input_space, output_space, self.__degree)

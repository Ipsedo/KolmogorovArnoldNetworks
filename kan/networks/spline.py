# -*- coding: utf-8 -*-
from typing import Dict

import torch as th

from .utils import ActivationFunction


def b_spline(
    x: th.Tensor, k: int, n: int, x_min: float = 0.0, x_max: float = 1.0
) -> th.Tensor:
    x = x.unsqueeze(-1)

    def __knots(_i: th.Tensor) -> th.Tensor:
        return _i / n * (x_max - x_min) + x_min

    i_s = th.arange(-k, n, device=x.device)

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

    return th.movedim(__b_spline(i_s, k), -1, 1)


class BSpline(ActivationFunction):
    def __init__(self, degree: int, grid_size: int) -> None:
        super().__init__()

        self.__degree = degree
        self.__grid_size = grid_size

    def forward(self, x: th.Tensor) -> th.Tensor:
        return b_spline(x, self.__degree, self.__grid_size)

    def get_size(self) -> int:
        return self.__grid_size + self.__degree

    @classmethod
    def from_dict(cls, options: Dict[str, str]) -> "ActivationFunction":
        assert (
            "degree" in options
        ), 'Must specify "degree", example : "-a degree=2"'
        assert (
            "grid_size" in options
        ), 'Must specify "grid_size", example : "-a grid_size=8"'

        return cls(
            int(options["degree"]),
            int(options["grid_size"]),
        )

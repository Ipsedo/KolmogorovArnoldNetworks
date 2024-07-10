# -*- coding: utf-8 -*-
from math import factorial
from typing import Dict

import torch as th

from .utils import ActivationFunction


def tcheb_coef(n: int) -> th.Tensor:
    coef = th.zeros(n + 1, n)
    for i in range(n):
        for k in range((i + 1) // 2 + 1):
            coef[i + 1 - 2 * k, i] = (
                (i + 1)
                / 2
                * (-1) ** k
                * 2 ** (i + 1 - 2 * k)
                * factorial(i - k)
                / factorial(k)
                / factorial(i + 1 - 2 * k)
            )

    return coef


# pylint: disable=duplicate-code
class Tchebychev(ActivationFunction):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.__n = n
        self._coef: th.Tensor
        self.register_buffer("_coef", tcheb_coef(n))

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = th.tanh(x.unsqueeze(-1).expand(*x.size(), self.__n + 1))
        out = th.pow(out, th.arange(0, self.__n + 1, device=x.device))
        out = th.einsum("...a,ab->...b", out, self._coef)
        return out.movedim(-1, 1)

    def get_size(self) -> int:
        return self.__n

    @classmethod
    def from_dict(cls, options: Dict[str, str]) -> "ActivationFunction":
        assert "n" in options, 'Must specify "n", example : "-a n=5"'

        return cls(int(options["n"]))

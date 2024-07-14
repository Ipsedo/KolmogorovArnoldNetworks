# -*- coding: utf-8 -*-
from math import factorial
from typing import Dict

import torch as th

from .activation import ActivationFunction, PolyCoefActivation


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
class Tchebychev(PolyCoefActivation):
    def __init__(self, n: int) -> None:
        super().__init__(n, tcheb_coef(n))

    @classmethod
    def from_dict(cls, options: Dict[str, str]) -> "ActivationFunction":
        assert "n" in options, 'Must specify "n", example : "-a n=5"'

        return cls(int(options["n"]))

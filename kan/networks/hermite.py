# -*- coding: utf-8 -*-
from math import factorial
from typing import Dict

import torch as th

from .utils import ActivationFunction


def __old_hermite(x: th.Tensor, n: int) -> th.Tensor:
    h_s = [th.ones(*x.size(), device=x.device), x]

    for i in range(1, n):
        h_s.append(x * h_s[i] - i * h_s[i - 1])

    return th.slice_copy(th.stack(h_s, dim=-1), -1, 1) / th.exp(
        th.lgamma(th.arange(2, n + 2, device=x.device)) / 2
    )


def scale_hermite(x_hermite: th.Tensor) -> th.Tensor:
    return th.exp(-(x_hermite**2) / 2)


def hermite(x: th.Tensor, n: int) -> th.Tensor:
    h_s = [th.ones(*x.size(), device=x.device), 2 * x]

    # Iteratively compute Hermite polynomials from 2 to n
    for k in range(2, n + 1):
        h_s.append(2 * x * h_s[k - 1] - 2 * (k - 1) * h_s[k - 2])

    return th.slice_copy(th.stack(h_s, dim=1), 1, 1)


def hermite_coef(n: int) -> th.Tensor:
    coef = th.zeros(n + 1, n)
    for i in range(n):
        for k in range((i + 1) // 2 + 1):
            coef[i + 1 - 2 * k, i] = (
                (-1) ** k
                / 2**k
                / factorial(k)
                / factorial(i + 1 - 2 * k)
                * th.exp(th.lgamma(th.tensor(i + 2)) / 2)
            )

    return coef


class Hermite(ActivationFunction):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.__n = n
        self._coef: th.Tensor
        self.register_buffer("_coef", hermite_coef(n))

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = x.unsqueeze(-1).expand(*x.size(), self.__n + 1)
        out = th.pow(out, th.arange(0, self.__n + 1, device=x.device))
        out = th.einsum("...a,ab->...b", out, self._coef)
        return out.movedim(-1, 1)

    def get_size(self) -> int:
        return self.__n

    @classmethod
    def from_dict(cls, options: Dict[str, str]) -> "ActivationFunction":
        assert "n" in options, 'Must specify "n", example : "-a n=5"'

        return cls(int(options["n"]))

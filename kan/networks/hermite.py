# -*- coding: utf-8 -*-
import torch as th

from .utils import ActivationFunction


def hermite(x: th.Tensor, n: int) -> th.Tensor:
    h_s = [th.ones(*x.size(), device=x.device), x]

    for i in range(1, n):
        h_s.append(x * h_s[i] - i * h_s[i - 1])

    return th.slice_copy(th.stack(h_s, dim=-1), -1, 1) / th.exp(
        th.lgamma(th.arange(2, n + 2, device=x.device)) / 2
    )


class Hermite(ActivationFunction):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.__n = n

        self._hermite_factors: th.Tensor
        self.register_buffer(
            "_hermite_factors", th.tensor(1e-1) ** th.arange(0, self.__n)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return hermite(x, self.__n) * self._hermite_factors

    def get_size(self) -> int:
        return self.__n

# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict

import torch as th
from torch import nn


class ActivationFunction(ABC, nn.Module):
    @abstractmethod
    def get_size(self) -> int:
        pass

    @abstractmethod
    def forward(self, x: th.Tensor) -> th.Tensor:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, options: Dict[str, str]) -> "ActivationFunction":
        pass


class PolyCoefActivation(ActivationFunction, ABC):
    def __init__(self, n: int, coefficients: th.Tensor) -> None:
        super().__init__()
        assert coefficients.size(0) == n + 1
        assert coefficients.size(1) == n

        self.__n = n
        self._coefficients: th.Tensor
        self.register_buffer("_coefficients", coefficients)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.einsum(
            "b...a,ac->bc...",
            th.pow(
                th.tanh(x.unsqueeze(-1)),
                th.arange(0, self.__n + 1, device=x.device),
            ),
            self._coefficients,
        )

    def get_size(self) -> int:
        return self.__n

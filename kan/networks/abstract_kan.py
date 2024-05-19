# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from statistics import mean
from typing import List, Tuple

import numpy as np
import torch as th
from torch import nn
from torch.nn.init import xavier_normal_


class AbstractKAN(ABC, nn.Module):
    def __init__(self, input_space: int, output_space: int) -> None:
        super().__init__()

        self.__w = nn.Parameter(th.randn(1, output_space, input_space))

        xavier_normal_(self.__w)

    @abstractmethod
    def _activation_function(self, x: th.Tensor) -> th.Tensor:
        pass

    @abstractmethod
    def _residual_activation_function(self, x: th.Tensor) -> th.Tensor:
        pass

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 2

        x = x.unsqueeze(1)  # broadcast output dim

        res_act_x = self._residual_activation_function(x)
        learned_act_x = self._activation_function(x)

        return th.sum(self.__w * (res_act_x + learned_act_x), dim=2)


class AbstractKanLayers(ABC, nn.Sequential):
    def __init__(self, layers: List[Tuple[int, int]]) -> None:
        super().__init__(*[self._get_layer(c_i, c_o) for c_i, c_o in layers])

    @abstractmethod
    def _get_layer(self, input_space: int, output_space: int) -> AbstractKAN:
        pass

    def count_parameters(self) -> int:
        return sum(int(np.prod(p.size())) for p in self.parameters())

    def grad_norm(self) -> float:
        return mean(float(p.norm().item()) for p in self.parameters())

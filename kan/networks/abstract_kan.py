# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List, Tuple

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
        assert len(x.size()) >= 2

        x = x.unsqueeze(1)

        res_act_x = self._residual_activation_function(x)
        learned_act_x = self._activation_function(x)

        assert len(res_act_x.size()) == len(x.size())
        assert len(learned_act_x.size()) == len(x.size())

        assert res_act_x.size(0) == x.size(0)
        assert res_act_x.size(2) == x.size(2)

        assert learned_act_x.size(0) == x.size(0)
        assert learned_act_x.size(1) == x.size(1)
        assert learned_act_x.size(2) == x.size(2)

        return th.sum(self.__w * (res_act_x + learned_act_x), dim=2)


class AbstractKanLayers(ABC, nn.Sequential):
    def __init__(self, layers: List[Tuple[int, int]]) -> None:
        super().__init__(*[self._get_layer(c_i, c_o) for c_i, c_o in layers])

    @abstractmethod
    def _get_layer(self, input_space: int, output_space: int) -> AbstractKAN:
        pass

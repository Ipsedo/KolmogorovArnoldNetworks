# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch as th
from torch import nn
from torch.nn.init import normal_, xavier_normal_

from .utils import BaseModule


class LinearKAN(nn.Module):
    def __init__(self, in_features: int, out_features: int, n: int) -> None:
        super().__init__()

        self.__in_features = in_features

        self.__w = nn.Parameter(th.ones(in_features, out_features))
        self.__c = nn.Parameter(th.ones(in_features, out_features, n))

        xavier_normal_(self.__w)
        normal_(self.__c, 0, 1e-1)

    @abstractmethod
    def _residual_act(self, x: th.Tensor) -> th.Tensor:
        pass

    @abstractmethod
    def _act(self, x: th.Tensor) -> th.Tensor:
        pass

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 2
        assert x.size(1) == self.__in_features

        # output dim
        x = x.unsqueeze(2)

        return th.sum(
            self.__w
            * (
                self._residual_act(x) + th.sum(self.__c * self._act(x), dim=-1)
            ),
            1,  # sum over input space
        )


class LinearKanLayers(ABC, nn.Sequential, BaseModule):
    def __init__(self, layers: List[Tuple[int, int]]) -> None:
        super().__init__(*[self._get_layer(c_i, c_o) for c_i, c_o in layers])

    @abstractmethod
    def _get_layer(self, input_space: int, output_space: int) -> LinearKAN:
        pass

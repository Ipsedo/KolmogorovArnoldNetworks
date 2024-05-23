# -*- coding: utf-8 -*-
from typing import Callable, List, Tuple

import torch as th
from torch import nn
from torch.nn.init import normal_, xavier_normal_

from .utils import ActivationFunction, InfoModule


class LinearKAN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_fun: ActivationFunction,
        res_act_fun: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        super().__init__()

        self.__in_features = in_features

        self.__act_fun = act_fun
        self.__res_act_fun = res_act_fun

        self.__w = nn.Parameter(th.ones(in_features, out_features))
        self.__c = nn.Parameter(
            th.ones(in_features, out_features, self.__act_fun.get_size())
        )

        xavier_normal_(self.__w)
        normal_(self.__c, 0, 1e-1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 2
        assert x.size(1) == self.__in_features

        # output dim
        x = x.unsqueeze(2)

        return th.sum(
            self.__w
            * (
                self.__res_act_fun(x)
                + th.sum(self.__c * self.__act_fun(x), dim=-1)
            ),
            1,  # sum over input space
        )


class LinearKanLayers(nn.Sequential, InfoModule):
    def __init__(
        self,
        layers: List[Tuple[int, int]],
        act_fun: ActivationFunction,
        res_act_fun: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        super().__init__(
            *[LinearKAN(c_i, c_o, act_fun, res_act_fun) for c_i, c_o in layers]
        )

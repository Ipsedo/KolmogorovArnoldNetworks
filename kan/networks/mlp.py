# -*- coding: utf-8 -*-
from typing import List, Tuple

from torch import nn

from .utils import BaseModule


class MLP(nn.Sequential, BaseModule):
    def __init__(self, layers: List[Tuple[int, int]]) -> None:
        super().__init__(
            *[
                nn.Sequential(
                    nn.Linear(c_i, c_o),
                    nn.Mish() if i != len(layers) - 1 else nn.Identity(),
                )
                for i, (c_i, c_o) in enumerate(layers)
            ]
        )

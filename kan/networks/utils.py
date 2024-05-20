# -*- coding: utf-8 -*-
from statistics import mean

import numpy as np
import torch as th
from torch import nn


class BaseModule(nn.Module):

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x

    def count_parameters(self) -> int:
        return sum(int(np.prod(p.size())) for p in self.parameters())

    def grad_norm(self) -> float:
        return mean(float(p.norm().item()) for p in self.parameters())

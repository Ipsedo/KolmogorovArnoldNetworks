# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from statistics import mean
from typing import Dict

import numpy as np
import torch as th
from torch import nn


class InfoModule(ABC, nn.Module):
    def count_parameters(self) -> int:
        return sum(int(np.prod(p.size())) for p in self.parameters())

    def grad_norm(self) -> float:
        return mean(float(p.norm().item()) for p in self.parameters())


class PrintSize(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        print(x.size())
        return x

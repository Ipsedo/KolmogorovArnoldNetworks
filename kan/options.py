# -*- coding: utf-8 -*-
from typing import Callable, Dict, Final, List, Literal, NamedTuple, Tuple

import torch as th
from torch.nn import functional as F

from .networks import (
    ActivationFunction,
    BaseModule,
    BSpline,
    Conv2dKanLayers,
    Hermite,
    LinearKanLayers,
)

# Models

ResidualActivation = Literal["relu", "lrelu", "gelu", "silu", "mish", "none"]


class LinearOptions(NamedTuple):
    layers: List[Tuple[int, int]]
    residual_activation: ResidualActivation


class ConvOptions(NamedTuple):
    channels: List[Tuple[int, int]]
    kernel_sizes: List[int]
    strides: List[int]
    paddings: List[int]
    residual_activation: ResidualActivation


# Activation functions


class HermiteOptions(NamedTuple):
    n_hermite: int


class SplineOptions(NamedTuple):
    degree: int
    grid_size: int


# Factory

_ACTIVATION_FUNCTIONS: Final[Dict[str, Callable[[th.Tensor], th.Tensor]]] = {
    "relu": F.relu,
    "lrelu": F.leaky_relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "mish": F.mish,
    "none": th.zeros_like,
}


class ModelOptions(NamedTuple):
    model_options: LinearOptions | ConvOptions
    activation_options: HermiteOptions | SplineOptions

    def get_model(self) -> BaseModule:
        act_fun: ActivationFunction
        if isinstance(self.activation_options, HermiteOptions):
            act_fun = Hermite(self.activation_options.n_hermite)
        elif isinstance(self.activation_options, SplineOptions):
            act_fun = BSpline(
                self.activation_options.degree,
                self.activation_options.grid_size,
            )
        else:
            raise ValueError(
                f"Unknown activation options: {self.activation_options}"
            )

        if isinstance(self.model_options, LinearOptions):
            return LinearKanLayers(
                self.model_options.layers,
                act_fun,
                _ACTIVATION_FUNCTIONS[self.model_options.residual_activation],
            )
        if isinstance(self.model_options, ConvOptions):
            return Conv2dKanLayers(
                self.model_options.channels,
                self.model_options.kernel_sizes,
                self.model_options.strides,
                self.model_options.paddings,
                act_fun,
                _ACTIVATION_FUNCTIONS[self.model_options.residual_activation],
            )
        raise ValueError(f"Unknown model options: {self.model_options}")


# Training / Eval stuff


class TrainOptions(NamedTuple):
    dataset_path: str
    batch_size: int
    learning_rate: float
    nb_epoch: int
    cuda: bool

# -*- coding: utf-8 -*-
from typing import (
    Callable,
    Dict,
    Final,
    List,
    Literal,
    NamedTuple,
    Tuple,
    get_args,
)

import torch as th
from torch.nn import functional as F

from .data import (
    ClassificationDataset,
    TensorCIFAR10,
    TensorCIFAR100,
    TensorImageNet,
    TensorMNIST,
)
from .networks import (
    ActivationFunction,
    BaseModule,
    BSpline,
    Conv2dKanLayers,
    Hermite,
)

# Models

_Activation = Literal["hermite", "b-spline"]
ResidualActivation = Literal["relu", "lrelu", "gelu", "silu", "mish", "none"]


class LinearOptions(NamedTuple):
    layers: List[Tuple[int, int]]
    residual_activation: ResidualActivation


class ConvOptions(NamedTuple):
    channels: List[Tuple[int, int]]
    kernel_sizes: List[int] | int
    strides: List[int] | int
    paddings: List[int] | int
    linear_sizes: List[Tuple[int, int]]
    residual_activation: ResidualActivation


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
    model_options: ConvOptions
    activation_compound_options: Tuple[str, Dict[str, str]]

    @staticmethod
    def __to_list(l_i: List[int] | int, length: int) -> List[int]:
        return l_i if isinstance(l_i, list) else [l_i] * length

    def get_model(self) -> BaseModule:
        act_fun_name, act_fun_options = self.activation_compound_options

        assert act_fun_name in get_args(_Activation)

        act_fun: ActivationFunction

        if act_fun_name == "hermite":
            act_fun = Hermite(int(act_fun_options["n"]))
        elif act_fun_name == "b-spline":
            act_fun = BSpline(
                int(act_fun_options["degree"]),
                int(act_fun_options["grid_size"]),
            )
        else:
            raise ValueError(f"Unknown activation options: {act_fun_name}")

        return Conv2dKanLayers(
            self.model_options.channels,
            self.__to_list(
                self.model_options.kernel_sizes,
                len(self.model_options.channels),
            ),
            self.__to_list(
                self.model_options.strides,
                len(self.model_options.channels),
            ),
            self.__to_list(
                self.model_options.paddings,
                len(self.model_options.channels),
            ),
            self.model_options.linear_sizes,
            act_fun,
            _ACTIVATION_FUNCTIONS[self.model_options.residual_activation],
        )


# Training / Eval stuff

DatasetName = Literal["cifar10", "cifar100", "mnist", "image-net"]


class TrainOptions(NamedTuple):
    dataset_path: str
    dataset: DatasetName
    train_ratio: float
    batch_size: int
    learning_rate: float
    nb_epoch: int
    cuda: bool

    def get_dataset(self) -> ClassificationDataset:
        if self.dataset == "cifar10":
            return TensorCIFAR10(self.dataset_path, train=True, download=True)
        if self.dataset == "cifar100":
            return TensorCIFAR100(self.dataset_path, train=True, download=True)
        if self.dataset == "mnist":
            return TensorMNIST(
                self.dataset_path, train=True, download=True, flatten=False
            )
        if self.dataset == "image-net":
            return TensorImageNet(self.dataset_path)
        raise ValueError(f"Unknown dataset {self.dataset}")

# -*- coding: utf-8 -*-
from typing import Callable, Dict, Final, List, Literal, NamedTuple, Set, Tuple

import torch as th
from torch.nn import functional as F

from .data import (
    ClassificationDataset,
    TensorCIFAR10,
    TensorCIFAR100,
    TensorImageNet,
    TensorMNIST,
)
from .networks import ActivationFunction, BSpline, Conv2dKanLayers, Hermite

# Models
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
_ACTIVATIONS: Final[Set[str]] = {"hermite", "b-spline"}
_RESIDUAL_ACTIVATIONS: Final[Dict[str, Callable[[th.Tensor], th.Tensor]]] = {
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

    def __to_list(self, l_i: List[int] | int) -> List[int]:
        return (
            l_i
            if isinstance(l_i, list)
            else [l_i] * len(self.model_options.channels)
        )

    def get_model(self) -> Conv2dKanLayers:
        act_fun_name, act_fun_options = self.activation_compound_options

        assert act_fun_name in _ACTIVATIONS

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
            self.__to_list(self.model_options.kernel_sizes),
            self.__to_list(self.model_options.strides),
            self.__to_list(self.model_options.paddings),
            self.model_options.linear_sizes,
            act_fun,
            _RESIDUAL_ACTIVATIONS[self.model_options.residual_activation],
        )


# Training / Eval stuff

DatasetName = Literal["cifar10", "cifar100", "mnist", "image-net"]


def _get_dataset(
    dataset: DatasetName, dataset_path: str
) -> ClassificationDataset:
    if dataset == "cifar10":
        return TensorCIFAR10(dataset_path, train=True, download=True)
    if dataset == "cifar100":
        return TensorCIFAR100(dataset_path, train=True, download=True)
    if dataset == "mnist":
        return TensorMNIST(
            dataset_path, train=True, download=True, flatten=False
        )
    if dataset == "image-net":
        return TensorImageNet(dataset_path)
    raise ValueError(f"Unknown dataset {dataset}")


class TrainOptions(NamedTuple):
    dataset_path: str
    dataset: DatasetName
    output_path: str
    train_ratio: float
    batch_size: int
    learning_rate: float
    nb_epoch: int
    save_every: int
    cuda: bool

    def get_dataset(self) -> ClassificationDataset:
        return _get_dataset(self.dataset, self.dataset_path)


class InferOptions(NamedTuple):
    dataset_path: str
    dataset: DatasetName
    batch_size: int
    model_state_dict_path: str
    output_csv_path: str
    cuda: bool

    def get_dataset(self) -> ClassificationDataset:
        return _get_dataset(self.dataset, self.dataset_path)

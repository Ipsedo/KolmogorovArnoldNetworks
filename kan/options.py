# -*- coding: utf-8 -*-
from typing import Callable, Dict, Final, List, NamedTuple, Tuple, Type

import torch as th
from torch.nn import functional as F

from .data import (
    ClassificationDataset,
    TensorCaltech256,
    TensorCIFAR10,
    TensorCIFAR100,
    TensorEuroSAT,
    TensorImageNet,
    TensorMNIST,
)
from .networks import ActivationFunction, BSpline, Conv2dKanLayers, Hermite

# Models


class LinearOptions(NamedTuple):
    layers: List[Tuple[int, int]]
    residual_activation: str


class ConvOptions(NamedTuple):
    channels: List[Tuple[int, int]]
    kernel_sizes: List[int] | int
    strides: List[int] | int
    paddings: List[int] | int
    linear_sizes: List[Tuple[int, int]]
    residual_activation: str


# Factory
_ACTIVATIONS: Final[Dict[str, Type[ActivationFunction]]] = {
    "hermite": Hermite,
    "b-spline": BSpline,
}

_RESIDUAL_ACTIVATIONS: Final[Dict[str, Callable[[th.Tensor], th.Tensor]]] = {
    "relu": F.relu,
    "lrelu": F.leaky_relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "mish": F.mish,
    "none": th.zeros_like,
}


def get_activation_names() -> List[str]:
    return list(_ACTIVATIONS.keys())


def get_residual_activation_names() -> List[str]:
    return list(_RESIDUAL_ACTIVATIONS.keys())


class ModelOptions(NamedTuple):
    conv_options: ConvOptions
    activation_compound_options: Tuple[str, Dict[str, str]]
    cuda: bool

    def __to_list(self, l_i: List[int] | int) -> List[int]:
        return (
            l_i
            if isinstance(l_i, list)
            else [l_i] * len(self.conv_options.channels)
        )

    def get_model(self) -> Conv2dKanLayers:
        act_fun_name, act_fun_options = self.activation_compound_options

        assert act_fun_name in _ACTIVATIONS, (
            f'Unrecognized activation function "{act_fun_name}". '
            f"Available activations : {'|'.join(_ACTIVATIONS.keys())}"
        )

        return Conv2dKanLayers(
            self.conv_options.channels,
            self.__to_list(self.conv_options.kernel_sizes),
            self.__to_list(self.conv_options.strides),
            self.__to_list(self.conv_options.paddings),
            self.conv_options.linear_sizes,
            _ACTIVATIONS[act_fun_name].from_dict(act_fun_options),
            _RESIDUAL_ACTIVATIONS[self.conv_options.residual_activation],
        )


# Training / Eval stuff

_DATASET_CONSTRUCTOR: Final[Dict[str, Type[ClassificationDataset]]] = {
    "cifar10": TensorCIFAR10,
    "cifar100": TensorCIFAR100,
    "mnist": TensorMNIST,
    "imagenet": TensorImageNet,
    "eurosat": TensorEuroSAT,
    "caltech": TensorCaltech256,
}


def get_dataset_names() -> List[str]:
    return list(_DATASET_CONSTRUCTOR.keys())


def _get_dataset(dataset: str, dataset_path: str) -> ClassificationDataset:
    assert dataset in _DATASET_CONSTRUCTOR, (
        f"Unrecognized dataset '{dataset}'. "
        f"Available datasets : {'|'.join(_DATASET_CONSTRUCTOR.keys())}"
    )

    return _DATASET_CONSTRUCTOR[dataset](dataset_path)


class TrainOptions(NamedTuple):
    dataset_path: str
    dataset: str
    output_path: str
    train_ratio: float
    batch_size: int
    learning_rate: float
    nb_epoch: int
    save_every: int

    def get_dataset(self) -> ClassificationDataset:
        return _get_dataset(self.dataset, self.dataset_path)


class InferOptions(NamedTuple):
    dataset_path: str
    dataset: str
    batch_size: int
    model_state_dict_path: str
    output_csv_path: str

    def get_dataset(self) -> ClassificationDataset:
        return _get_dataset(self.dataset, self.dataset_path)

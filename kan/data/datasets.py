# -*- coding: utf-8 -*-

import torch as th
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.transforms import Compose, ToTensor

from .transform import Flatten, MinMaxNorm, ToDType


class TensorMNIST(MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        flatten: bool = False,
        download: bool = False,
    ):
        transforms = [
            ToTensor(),
            ToDType(th.float),
            MinMaxNorm(0.0, 255.0),
        ]

        if flatten:
            transforms.append(Flatten(0, -1))

        compose = Compose(transforms)

        super().__init__(
            root,
            train,
            transform=compose,
            target_transform=None,
            download=download,
        )


class TensorCIFAR100(CIFAR100):
    def __init__(self, root: str, train: bool = True, download: bool = False):
        transforms = [
            ToTensor(),
            ToDType(th.float),
        ]

        compose = Compose(transforms)

        super().__init__(
            root,
            train,
            transform=compose,
            target_transform=None,
            download=download,
        )


class TensorCIFAR10(CIFAR10):
    def __init__(self, root: str, train: bool = True, download: bool = False):
        transforms = [
            ToTensor(),
            ToDType(th.float),
        ]

        compose = Compose(transforms)

        super().__init__(
            root,
            train,
            transform=compose,
            target_transform=None,
            download=download,
        )

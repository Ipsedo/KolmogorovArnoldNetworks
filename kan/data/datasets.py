# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict

import torch as th
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from .transform import Flatten, MinMaxNorm, ToDType


class ClassificationDataset(ABC, Dataset):
    @abstractmethod
    def get_class_nb(self) -> int:
        pass

    @abstractmethod
    def get_class_to_idx(self) -> Dict[str, int]:
        pass


class TensorMNIST(MNIST, ClassificationDataset):
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

    def get_class_nb(self) -> int:
        return len(self.class_to_idx)

    def get_class_to_idx(self) -> Dict[str, int]:
        class_to_idx: Dict[str, int] = self.class_to_idx
        return class_to_idx


class TensorCIFAR100(CIFAR100, ClassificationDataset):
    def __init__(
        self, root: str, train: bool = True, download: bool = False
    ) -> None:

        super().__init__(
            root,
            train,
            transform=Compose(
                [
                    ToTensor(),
                    ToDType(th.float),
                ]
            ),
            target_transform=None,
            download=download,
        )

    def get_class_nb(self) -> int:
        return len(self.class_to_idx)

    def get_class_to_idx(self) -> Dict[str, int]:
        class_to_idx: Dict[str, int] = self.class_to_idx
        return class_to_idx


class TensorCIFAR10(CIFAR10, ClassificationDataset):
    def __init__(
        self, root: str, train: bool = True, download: bool = False
    ) -> None:
        super().__init__(
            root,
            train,
            transform=Compose(
                [
                    ToTensor(),
                    ToDType(th.float),
                ]
            ),
            target_transform=None,
            download=download,
        )

    def get_class_nb(self) -> int:
        return len(self.class_to_idx)

    def get_class_to_idx(self) -> Dict[str, int]:
        class_to_idx: Dict[str, int] = self.class_to_idx
        return class_to_idx


class TensorImageNet(ImageFolder, ClassificationDataset):
    def __init__(self, root: str):
        super().__init__(
            root,
            transform=Compose(
                [
                    ToTensor(),
                    ToDType(th.float),
                    Resize(256),
                    CenterCrop(224),
                ]
            ),
            target_transform=None,
        )

    def get_class_nb(self) -> int:
        return len(self.class_to_idx)

    def get_class_to_idx(self) -> Dict[str, int]:
        class_to_idx: Dict[str, int] = self.class_to_idx
        return class_to_idx

# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch as th
from torch.utils.data import Dataset
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    MNIST,
    Caltech256,
    EuroSAT,
    ImageFolder,
)
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from .transform import Flatten, MinMaxNorm, RangeChange, ToDType, ToRGB

# pylint: disable=too-many-ancestors


class ClassificationDataset(ABC, Dataset):
    def __init__(self, root: str, **kwargs: Any) -> None:
        pass

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
            train=train,
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
        self, root: str, train: bool = True, download: bool = True
    ) -> None:

        super().__init__(
            root,
            train=train,
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
        self, root: str, train: bool = True, download: bool = True
    ) -> None:
        super().__init__(
            root,
            train=train,
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
                    RangeChange(-1.0, 1.0),
                ]
            ),
            target_transform=None,
        )

    def get_class_nb(self) -> int:
        return len(self.class_to_idx)

    def get_class_to_idx(self) -> Dict[str, int]:
        class_to_idx: Dict[str, int] = self.class_to_idx
        return class_to_idx


class TensorEuroSAT(EuroSAT, ClassificationDataset):
    def __init__(self, root: str, download: bool = True) -> None:
        super().__init__(
            root=root,
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


class TensorCaltech256(Caltech256, ClassificationDataset):
    def __init__(self, root: str, download: bool = True) -> None:
        super().__init__(
            root=root,
            transform=Compose(
                [
                    ToTensor(),
                    ToDType(th.float),
                    Resize(600),
                    CenterCrop(512),
                    ToRGB(),
                ]
            ),
            target_transform=None,
            download=download,
        )

    def get_class_nb(self) -> int:
        return len(self.categories)

    def get_class_to_idx(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.categories)}

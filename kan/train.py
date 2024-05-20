# -*- coding: utf-8 -*-
from typing import List, NamedTuple, Tuple

import torch as th
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification.precision_recall import (
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from .data import MinMaxNorm, ToDType
from .networks import HermiteConv2dKanLayers


class SplineKanOptions(NamedTuple):
    layers: List[Tuple[int, int]]
    degree: int
    grid_size: int


class HermiteKanOptions(NamedTuple):
    layers: List[Tuple[int, int]]
    n_hermite: int


class HermiteConv2dKanOptions(NamedTuple):
    channels: List[Tuple[int, int]]
    n_hermite: int
    kernel_size: int
    stride: int
    padding: int


class TrainOptions(NamedTuple):
    dataset_path: str
    batch_size: int
    learning_rate: float
    nb_epoch: int
    cuda: bool


def train(
    kan_options: HermiteConv2dKanOptions, train_options: TrainOptions
) -> None:
    # pylint: disable=too-many-locals

    # model = SplineKanLayers(
    #     kan_options.layers, kan_options.degree, kan_options.grid_size
    # )

    # model = MLP(kan_options.layers)

    # model = HermiteKanLayers(kan_options.layers, kan_options.n_hermite)

    model = HermiteConv2dKanLayers(
        kan_options.channels,
        kan_options.n_hermite,
        kan_options.kernel_size,
        kan_options.stride,
        kan_options.padding,
    )

    optim = th.optim.Adam(model.parameters(), lr=train_options.learning_rate)

    print("parameters :", model.count_parameters())

    data_transform = Compose(
        [
            ToTensor(),
            ToDType(th.float),
            MinMaxNorm(0.0, 255.0),
            # Flatten(0, -1),
        ]
    )

    train_dataloader = DataLoader(
        CIFAR100(
            train_options.dataset_path,
            train=True,
            download=True,
            transform=data_transform,
        ),
        train_options.batch_size,
        shuffle=True,
        num_workers=6,
    )

    test_dataloader = DataLoader(
        CIFAR100(
            train_options.dataset_path,
            train=False,
            download=True,
            transform=data_transform,
        ),
        train_options.batch_size,
        shuffle=True,
        num_workers=6,
    )

    if train_options.cuda:
        model.cuda()
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    for e in range(train_options.nb_epoch):

        train_precision = MulticlassPrecision(num_classes=100).to(device)
        train_recall = MulticlassRecall(num_classes=100).to(device)

        train_tqdm_bar = tqdm(train_dataloader)

        model.train()

        for x, y in train_tqdm_bar:
            x = x.to(device)
            y = y.to(device)

            o = model(x)

            _ = train_precision(o, y), train_recall(o, y)

            loss = F.cross_entropy(o, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_tqdm_bar.set_description(
                f"Epoch {e} / {train_options.nb_epoch}, "
                f"loss = {loss.item():.4f}, "
                f"prec = {train_precision.compute().item():.4f}, "
                f"rec = {train_recall.compute().item():.4f}, "
                f"grad_norm = {model.grad_norm():.4f}"
            )

        # Test
        with th.no_grad():

            test_losses = []
            test_precision = MulticlassPrecision(num_classes=100).to(device)
            test_recall = MulticlassRecall(num_classes=100).to(device)

            test_tqdm_bar = tqdm(test_dataloader)

            model.eval()

            for x, y in test_tqdm_bar:
                x = x.to(device)
                y = y.to(device)

                o = model(x)
                _ = test_precision(o, y), test_recall(o, y)

                test_losses.append(F.cross_entropy(o, y, reduction="none"))

                test_tqdm_bar.set_description(
                    f"Test loss epoch {e} : loss_mean = "
                    f"{th.mean(th.cat(test_losses, dim=0)):.4f}, "
                    f"prec = {test_precision.compute().item():.4f}, "
                    f"rec = {test_recall.compute().item():.4f}"
                )

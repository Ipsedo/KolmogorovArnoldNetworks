# -*- coding: utf-8 -*-
from typing import List, NamedTuple, Tuple

import torch as th
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from .networks import SplineKanLayers


class SplineKanOptions(NamedTuple):
    layers: List[Tuple[int, int]]
    degree: int
    grid_size: int


class TrainOptions(NamedTuple):
    dataset_path: str
    batch_size: int
    learning_rate: float
    nb_epoch: int
    cuda: bool


def train(kan_options: SplineKanOptions, train_options: TrainOptions) -> None:
    kan = SplineKanLayers(
        kan_options.layers, kan_options.degree, kan_options.grid_size
    )
    optim = th.optim.Adam(kan.parameters(), lr=train_options.learning_rate)

    train_dataloader = DataLoader(
        MNIST(
            train_options.dataset_path,
            train=True,
            download=True,
            transform=ToTensor(),
        ),
        train_options.batch_size,
        shuffle=True,
        num_workers=6,
    )

    test_dataloader = DataLoader(
        MNIST(
            train_options.dataset_path,
            train=False,
            download=True,
            transform=ToTensor(),
        ),
        train_options.batch_size,
        shuffle=True,
        num_workers=6,
    )

    if train_options.cuda:
        kan.cuda()
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    for e in range(train_options.nb_epoch):

        train_tqdm_bar = tqdm(train_dataloader)

        kan.train()

        for x, y in train_tqdm_bar:
            x = x.to(device).flatten(1, -1)
            y = y.to(device)

            o = kan(x)

            loss = F.cross_entropy(o, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_tqdm_bar.set_description(
                f"Epoch {e} / {train_options.nb_epoch}, "
                f"loss = {loss.item():.4f}"
            )

        # Test
        with th.no_grad():
            kan.eval()

            test_losses = []
            test_tqdm_bar = tqdm(test_dataloader)
            for x, y in test_tqdm_bar:
                x = x.to(device).flatten(1, -1)
                y = y.to(device)

                o = kan(x)

                test_losses.append(F.cross_entropy(o, y, reduction="none"))

                test_tqdm_bar.set_description(
                    f"Test loss epoch {e} : loss_mean = "
                    f"{th.mean(th.cat(test_losses, dim=0)):.4f}"
                )

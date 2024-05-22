# -*- coding: utf-8 -*-

import torch as th
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification.precision_recall import (
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm

from .data import TensorCIFAR10
from .options import ModelOptions, TrainOptions


def train(kan_options: ModelOptions, train_options: TrainOptions) -> None:
    # pylint: disable=too-many-locals

    model = kan_options.get_model()

    optim = th.optim.Adam(model.parameters(), lr=train_options.learning_rate)

    print("parameters :", model.count_parameters())

    train_dataloader = DataLoader(  # type: ignore
        TensorCIFAR10(
            train_options.dataset_path,
            train=True,
            download=True,
        ),
        train_options.batch_size,
        shuffle=True,
        num_workers=6,
    )

    test_dataloader = DataLoader(  # type: ignore
        TensorCIFAR10(
            train_options.dataset_path,
            train=False,
            download=True,
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

        train_precision = MulticlassPrecision(num_classes=10).to(device)
        train_recall = MulticlassRecall(num_classes=10).to(device)

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
            test_precision = MulticlassPrecision(num_classes=10).to(device)
            test_recall = MulticlassRecall(num_classes=10).to(device)

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

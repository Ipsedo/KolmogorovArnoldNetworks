# -*- coding: utf-8 -*-
import torch as th
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .metrics import PrecisionRecall
from .options import ModelOptions, TrainOptions


def train(kan_options: ModelOptions, train_options: TrainOptions) -> None:
    # pylint: disable=too-many-locals

    if train_options.cuda:
        th.backends.cudnn.benchmark = True

    model = kan_options.get_model()

    optim = th.optim.Adam(model.parameters(), lr=train_options.learning_rate)

    print("parameters :", model.count_parameters())

    dataset = train_options.get_dataset()
    train_dataset, eval_dataset = random_split(  # type: ignore
        dataset,
        [train_options.train_ratio, 1.0 - train_options.train_ratio],
    )

    train_dataloader = DataLoader(  # type: ignore
        train_dataset,
        train_options.batch_size,
        shuffle=True,
        num_workers=6,
    )

    test_dataloader = DataLoader(  # type: ignore
        eval_dataset,
        train_options.batch_size,
        shuffle=True,
        num_workers=6,
    )

    if train_options.cuda:
        model.cuda()
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    train_metric = PrecisionRecall(dataset.get_class_nb(), 128)

    for e in range(train_options.nb_epoch):

        train_tqdm_bar = tqdm(train_dataloader)

        model.train()

        for x, y in train_tqdm_bar:
            x = x.to(device)
            y = y.to(device)

            o = model(x)
            loss = F.cross_entropy(o, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_metric.add(F.softmax(o, -1), y)
            prec, rec = train_metric.get()

            train_tqdm_bar.set_description(
                f"Epoch {e} / {train_options.nb_epoch}, "
                f"loss = {loss.item():.4f}, "
                f"prec = {prec:.4f}, "
                f"rec = {rec:.4f}, "
                f"grad_norm = {model.grad_norm():.4f}"
            )

        # Test
        with th.no_grad():

            test_losses = []
            test_metric = PrecisionRecall(dataset.get_class_nb(), None)

            test_tqdm_bar = tqdm(test_dataloader)

            model.eval()

            for x, y in test_tqdm_bar:
                x = x.to(device)
                y = y.to(device)

                o = model(x)

                test_losses.append(
                    F.cross_entropy(F.softmax(o, -1), y, reduction="none")
                )

                test_metric.add(o, y)
                prec, rec = test_metric.get()

                test_tqdm_bar.set_description(
                    f"Test loss epoch {e} : loss_mean = "
                    f"{th.mean(th.cat(test_losses, dim=0)):.4f}, "
                    f"prec = {prec:.4f}, "
                    f"rec = {rec:.4f}"
                )

# -*- coding: utf-8 -*-
from os import makedirs
from os.path import exists, isdir, join

import torch as th
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .infer import infer_on_dataset
from .metrics import PrecisionRecall
from .options import ModelOptions, TrainOptions


def train(kan_options: ModelOptions, train_options: TrainOptions) -> None:
    # pylint: disable=too-many-locals,too-many-statements

    if not exists(train_options.output_path):
        makedirs(train_options.output_path)
    elif not isdir(train_options.output_path):
        raise NotADirectoryError(
            f"{train_options.output_path} is not a directory"
        )

    if train_options.cuda:
        th.backends.cudnn.benchmark = True

    model = kan_options.get_model()

    optim = th.optim.Adam(model.parameters(), lr=train_options.learning_rate)

    dataset = train_options.get_dataset()
    train_dataset, eval_dataset = random_split(  # type: ignore
        dataset,
        [train_options.train_ratio, 1.0 - train_options.train_ratio],
    )

    print("parameters :", model.count_parameters())
    print("train dataset :", len(train_dataset))
    print("eval dataset :", len(eval_dataset))

    train_dataloader = DataLoader(  # type: ignore
        train_dataset,
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
    idx = 0

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
                f"grad_norm = {model.grad_norm():.4f}, "
                f"save : [{idx % train_options.save_every:04d} "
                f"/ {train_options.save_every}]"
            )

            if idx % train_options.save_every == 0:
                th.save(
                    model.state_dict(),
                    join(train_options.output_path, f"model_{idx}.pt"),
                )

                th.save(
                    optim.state_dict(),
                    join(train_options.output_path, f"optim_{idx}.pt"),
                )

            idx += 1

        # Test
        _ = infer_on_dataset(
            model,
            eval_dataset,
            train_options.batch_size,
            dataset.get_class_nb(),
            device,
        )

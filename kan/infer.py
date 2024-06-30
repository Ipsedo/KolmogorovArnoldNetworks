# -*- coding: utf-8 -*-
from typing import Tuple

import pandas as pd
import torch as th
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .metrics import PrecisionRecall
from .networks import Conv2dKanLayers
from .options import InferOptions, ModelOptions


def infer_on_dataset(
    model: Conv2dKanLayers,
    dataset: Dataset,
    batch_size: int,
    class_nb: int,
    device: th.device,
) -> Tuple[th.Tensor, th.Tensor]:
    # pylint: disable=too-many-locals
    with th.no_grad():
        test_dataloader = DataLoader(  # type: ignore
            dataset,
            batch_size,
            shuffle=False,
            num_workers=8,
        )

        test_metric = PrecisionRecall(class_nb, None)

        test_tqdm_bar = tqdm(test_dataloader)

        model.eval()

        predictions = []
        targets = []

        for x, y in test_tqdm_bar:
            x = x.to(device)
            y = y.to(device)

            o = model(x)

            predictions.append(o)
            targets.append(y)

            test_metric.add(F.softmax(o, -1), y)
            prec, rec = test_metric.get()

            test_tqdm_bar.set_description(
                f"Eval : " f"prec = {prec:.4f}, " f"rec = {rec:.4f}"
            )

        return th.cat(predictions, dim=0), th.cat(targets, dim=0)


def infer(model_options: ModelOptions, infer_options: InferOptions) -> None:
    dataset = infer_options.get_dataset()

    model = model_options.get_model()

    device = th.device("cuda") if model_options.cuda else th.device("cpu")
    model.to(device)

    model.load_state_dict(
        th.load(infer_options.model_state_dict_path, map_location=device)
    )

    predictions, targets = infer_on_dataset(
        model,
        dataset,
        infer_options.batch_size,
        dataset.get_class_nb(),
        device,
    )

    predicted_proba = F.softmax(predictions, dim=-1)

    df = pd.DataFrame(
        {
            "id": th.arange(predictions.size(0)).numpy().tolist(),
            "predicted_proba": [
                ",".join([str(f) for f in p.cpu().numpy().tolist()])
                for p in predicted_proba
            ],
            "targets": targets.cpu().numpy().tolist(),
        }
    )

    df.to_csv(infer_options.output_csv_path, sep=";", header=True, index=False)

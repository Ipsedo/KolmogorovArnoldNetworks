# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, TypeVar

import torch as th
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall

T = TypeVar("T")


class Metric(ABC, Generic[T]):
    def __init__(self, window_size: int | None = None) -> None:
        self.__window_size = window_size

        self.__window: List[Tuple[th.Tensor, ...]] = []

    def add(self, *args: th.Tensor) -> None:
        self.__window.append(tuple(a.detach() for a in args))

        if self.__window_size is not None:
            while len(self.__window) > self.__window_size:
                self.__window.pop(0)

    def get(self) -> T:
        return self._process_args(self.__window)

    @abstractmethod
    def _process_args(self, window: List[Tuple[th.Tensor, ...]]) -> T:
        pass


class PrecisionRecall(Metric):
    def __init__(self, n_class: int, window_size: int | None = None) -> None:
        super().__init__(window_size)

        self.__n_class = n_class

    def _process_args(
        self, window: List[Tuple[th.Tensor, ...]]
    ) -> Tuple[float, float]:
        out_l, pred_l = zip(*window)
        out, pred = th.cat(out_l, dim=0), th.cat(pred_l, dim=0)

        precision = MulticlassPrecision(self.__n_class).to(out.device)
        recall = MulticlassRecall(self.__n_class).to(out.device)

        return precision(out, pred).item(), recall(out, pred).item()

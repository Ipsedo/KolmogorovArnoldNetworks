# -*- coding: utf-8 -*-
from typing import Generic, TypeVar, Union

import torch as th

Number = TypeVar("Number", bound=Union[int, float])


class MinMaxNorm(Generic[Number]):
    def __init__(
        self, min_value: Number, max_value: Number, eps: float = 1e-8
    ) -> None:
        self.__min_value = float(min_value)
        self.__max_value = float(max_value)
        self.__eps = eps

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return (x - self.__min_value) / (
            self.__max_value - self.__min_value + self.__eps
        )


class ToDType:
    def __init__(self, dtype: th.dtype) -> None:
        self.__dtype = dtype

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return x.to(self.__dtype)

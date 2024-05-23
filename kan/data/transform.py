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


class Flatten:
    def __init__(self, start_dim: int, end_dim: int) -> None:
        self.__start_dim = start_dim
        self.__end_dim = end_dim

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return x.flatten(self.__start_dim, self.__end_dim)


class ToRGB:
    def __call__(self, x: th.Tensor) -> th.Tensor:
        if len(x.size()) == 2:
            return x.unsqueeze(0).repeat(3, 1, 1)
        if len(x.size()) == 3 and x.size(0) == 1:
            return x.repeat(3, 1, 1)
        return x

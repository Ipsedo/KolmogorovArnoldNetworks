# -*- coding: utf-8 -*-
from typing import List, Tuple

import pytest
import torch as th

from kan.networks import BSpline, LinearKAN, LinearKanLayers


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("features", [5, 6, 7])
@pytest.mark.parametrize("out", [1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("grid_size", [12, 13, 14])
def test_spline_kan(
    batch_size: int, features: int, out: int, degree: int, grid_size: int
) -> None:
    spline_kan = LinearKAN(
        features, out, BSpline(degree, grid_size), lambda t: t
    )
    x = th.randn(batch_size, features)

    o = spline_kan(x)

    assert len(o.size()) == len(x.size())
    assert o.size(0) == batch_size
    assert o.size(1) == out


@pytest.mark.parametrize(
    "layers", [[(16, 10), (10, 3)], [(2, 3)], [(16, 32), (32, 16), (16, 1)]]
)
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("grid_size", [12, 13, 14])
def test_spline_kan_layers(
    layers: List[Tuple[int, int]], batch_size: int, degree: int, grid_size: int
) -> None:
    kan_layers = LinearKanLayers(
        layers, BSpline(degree, grid_size), lambda t: t
    )

    input_space = layers[0][0]
    output_space = layers[-1][1]

    x = th.randn(batch_size, input_space)
    o = kan_layers(x)

    assert len(o.size()) == 2
    assert o.size(0) == batch_size
    assert o.size(1) == output_space

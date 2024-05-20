# -*- coding: utf-8 -*-
from typing import List, Tuple

import pytest
import torch as th

from kan.networks.hermite import HermiteKAN, HermiteKanLayers, hermite


@pytest.mark.parametrize("sizes", [(10, 3), (2,), (3, 9, 5)])
@pytest.mark.parametrize("n_hermite", [2, 3, 4])
def test_hermite(sizes: Tuple[int, ...], n_hermite: int) -> None:
    x = th.randn(*sizes)
    h = hermite(x, n_hermite)

    assert len(h.size()) == len(sizes) + 1
    assert all(h.size(i) == sizes[i] for i in range(len(sizes)))
    assert h.size(-1) == n_hermite


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("features", [5, 6, 7])
@pytest.mark.parametrize("out", [1, 2, 3])
@pytest.mark.parametrize("n_hermite", [1, 2, 3])
def test_hermite_kan(
    batch_size: int, features: int, out: int, n_hermite: int
) -> None:
    hermite_kan = HermiteKAN(features, out, n_hermite)
    hermite_kan.eval()

    x = th.randn(batch_size, features)

    o = hermite_kan(x)

    assert len(o.size()) == len(x.size())
    assert o.size(0) == batch_size
    assert o.size(1) == out


@pytest.mark.parametrize(
    "layers", [[(16, 10), (10, 3)], [(2, 3)], [(16, 32), (32, 16), (16, 1)]]
)
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("n_hermite", [1, 2, 3])
def test_hermite_kan_layers(
    layers: List[Tuple[int, int]], batch_size: int, n_hermite: int
) -> None:
    kan_layers = HermiteKanLayers(layers, n_hermite)
    kan_layers.eval()

    input_space = layers[0][0]
    output_space = layers[-1][1]

    x = th.randn(batch_size, input_space)
    o = kan_layers(x)

    assert len(o.size()) == 2
    assert o.size(0) == batch_size
    assert o.size(1) == output_space

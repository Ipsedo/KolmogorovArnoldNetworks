# -*- coding: utf-8 -*-
from typing import List, Tuple

import pytest
import torch as th

from kan.networks.mlp import MLP


@pytest.mark.parametrize(
    "layers", [[(16, 10), (10, 3)], [(2, 3)], [(16, 32), (32, 16), (16, 1)]]
)
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_mlp(layers: List[Tuple[int, int]], batch_size: int) -> None:
    kan_layers = MLP(layers)

    input_space = layers[0][0]
    output_space = layers[-1][1]

    x = th.randn(batch_size, input_space)
    o = kan_layers(x)

    assert len(o.size()) == 2
    assert o.size(0) == batch_size
    assert o.size(1) == output_space

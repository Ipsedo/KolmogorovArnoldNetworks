# -*- coding: utf-8 -*-
from typing import List, Tuple

import pytest
import torch as th

from kan.networks import Conv2dKan, Conv2dKanLayers, Hermite


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("in_channels", [1, 2, 3])
@pytest.mark.parametrize("out_channels", [1, 2, 3])
@pytest.mark.parametrize("sizes", [(10, 4), (4, 4), (8, 4)])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("n_hermite", [2, 3, 4])
def test_hermite_conv2d_kan(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    sizes: Tuple[int, int],
    kernel_size: int,
    stride: int,
    n_hermite: int,
) -> None:
    hermite_conv_kan = Conv2dKan(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        kernel_size // 2,
        Hermite(n_hermite),
        lambda t: t,
    )
    hermite_conv_kan.eval()

    x = th.randn(batch_size, in_channels, sizes[0], sizes[1])

    o = hermite_conv_kan(x)

    assert len(o.size()) == len(x.size())
    assert o.size(0) == batch_size
    assert o.size(1) == out_channels
    assert o.size(2) == sizes[0] // stride
    assert o.size(3) == sizes[1] // stride


@pytest.mark.parametrize(
    "channels",
    [[(16, 10), (10, 3)], [(2, 3), (3, 2)], [(16, 32), (32, 16)]],
)
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("n_hermite", [1, 2, 3])
def test_hermite_conv_kan_layers(
    channels: List[Tuple[int, int]], batch_size: int, n_hermite: int
) -> None:

    input_space = channels[0][0]
    output_space = channels[-1][1]

    conv_kan_layers = Conv2dKanLayers(
        channels,
        [3] * len(channels),
        [2] * len(channels),
        [1] * len(channels),
        [(output_space * 2 * 2, output_space)],
        Hermite(n_hermite),
        lambda t: t,
    )
    conv_kan_layers.eval()

    x = th.randn(batch_size, input_space, 8, 8)
    o = conv_kan_layers(x)

    assert len(o.size()) == 2
    assert o.size(0) == batch_size
    assert o.size(1) == output_space

# -*- coding: utf-8 -*-
import pytest
import torch as th

from kan.networks.spline import SplineKAN, b_spline


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("features", [5, 6, 7])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_b_spline(batch_size: int, features: int, degree: int) -> None:
    x = th.randn(batch_size, features)
    o = b_spline(x, degree)

    assert len(o.size()) == 2
    assert o.size(0) == batch_size
    assert o.size(1) == features


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("features", [5, 6, 7])
@pytest.mark.parametrize("out", [1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_spline_kan(
    batch_size: int, features: int, out: int, degree: int
) -> None:
    spline_kan = SplineKAN(features, out, degree)
    x = th.randn(batch_size, features)

    o = spline_kan(x)

    assert len(o.size()) == len(x.size())
    assert o.size(0) == batch_size
    assert o.size(1) == out

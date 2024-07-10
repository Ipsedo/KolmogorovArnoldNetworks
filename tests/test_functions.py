# -*- coding: utf-8 -*-
from typing import Tuple

import pytest
import torch as th

from kan.networks import Tchebychev
from kan.networks.hermite import Hermite
from kan.networks.spline import b_spline


@pytest.mark.parametrize("sizes", [(10, 3), (2,), (3, 9, 5)])
@pytest.mark.parametrize("n_hermite", [1, 2, 3])
def test_hermite(sizes: Tuple[int, ...], n_hermite: int) -> None:
    x = th.randn(*sizes)
    h = Hermite(n_hermite)
    o = h(x)

    assert len(o.size()) == len(sizes) + 1
    assert o.size(1) == n_hermite
    assert all(
        o.size(i + 1 if i >= 1 else 0) == s for i, s in enumerate(sizes)
    )


@pytest.mark.parametrize("sizes", [(10, 3), (2,), (3, 9, 5)])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_tchebychev(sizes: Tuple[int, ...], n: int) -> None:
    x = th.randn(*sizes)
    t = Tchebychev(n)
    o = t(x)

    assert len(o.size()) == len(sizes) + 1
    assert o.size(1) == n
    assert all(
        o.size(i + 1 if i >= 1 else 0) == s for i, s in enumerate(sizes)
    )


@pytest.mark.parametrize("sizes", [(10, 3), (2,), (3, 9, 5)])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("grid_size", [1, 2, 3])
def test_b_spline(sizes: Tuple[int, ...], degree: int, grid_size: int) -> None:
    x = th.randn(*sizes)
    o = b_spline(x, degree, grid_size)

    assert len(o.size()) == len(sizes) + 1
    assert o.size(1) == degree + grid_size
    assert all(
        o.size(i + 1 if i >= 1 else 0) == s for i, s in enumerate(sizes)
    )

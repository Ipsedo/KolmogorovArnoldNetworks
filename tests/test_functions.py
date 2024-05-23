# -*- coding: utf-8 -*-
from typing import Tuple

import pytest
import torch as th

from kan.networks.hermite import hermite
from kan.networks.spline import b_spline


@pytest.mark.parametrize("sizes", [(10, 3), (2,), (3, 9, 5)])
@pytest.mark.parametrize("n_hermite", [1, 2, 3])
def test_hermite(sizes: Tuple[int, ...], n_hermite: int) -> None:
    x = th.randn(*sizes)
    h = hermite(x, n_hermite)

    assert len(h.size()) == len(sizes) + 1
    assert all(h.size(i) == s for i, s in enumerate(sizes))
    assert h.size(-1) == n_hermite


@pytest.mark.parametrize("sizes", [(10, 3), (2,), (3, 9, 5)])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("grid_size", [1, 2, 3])
def test_b_spline(sizes: Tuple[int, ...], degree: int, grid_size: int) -> None:
    x = th.randn(*sizes)
    o = b_spline(x, degree, grid_size)

    assert len(o.size()) == len(sizes) + 1
    assert all(o.size(i) == s for i, s in enumerate(sizes))
    assert o.size(-1) == degree + grid_size

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Bare non-affine index READ (``a[i * i]``, no modular wrap) vectorizes as a per-lane GATHER.

``i**2`` classifies ``PerDimKind.GATHER`` (see
``analysis/test_tile_access_classifier.py::test_non_affine_iter_var_expression_classifies_gather``):
the index has no integer lane stride, so the tile path materializes the per-lane index tile
``[(i+0)**2, .., (i+W-1)**2]`` and gathers. This is the wrap-free counterpart of
``test_modular_gather_read.py``'s ``(i*i) % S`` kernel and pins the classifier's
GATHER contract end-to-end numerically.
"""
import numpy
import pytest

import dace
from tests.passes.vectorization.helpers.harness import S, X, run_vectorization_test

pytestmark = pytest.mark.tile_nodes


@dace.program
def square_index_gather_read_1d(out: dace.float64[X], a: dace.float64[S]):
    # ``i * i`` is non-affine (GATHER); S >= (X-1)**2 + 1 keeps every lane in bounds.
    for i, in dace.map[0:X:1]:
        out[i] = a[i * i]


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
@pytest.mark.parametrize("branch_mode", ["merge", "fp_factor"])
def test_square_index_gather_read_1d(branch_mode, remainder_strategy):
    """``out[i] = a[i * i]`` -> per-lane gather ``a[(i+l)**2]``; X=20 not a
    multiple of 8 -> remainder tile."""
    xv = 20
    sv = (xv - 1)**2 + 1
    run_vectorization_test(
        dace_func=square_index_gather_read_1d,
        arrays={"out": numpy.zeros(xv), "a": numpy.random.random(sv)},
        params={"X": xv, "S": sv},
        vector_width=8,
        sdfg_name="square_index_gather_read_1d",
        branch_mode=branch_mode,
        remainder_strategy=remainder_strategy,
    )

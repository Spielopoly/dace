# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests: vectorizer transients must never land in DaCe's reserved
``__return*`` namespace (bug 13).

``InsertTileLoadStore`` mints bridge transients from the staged data name
(``f"{an.data}_tile_out"`` etc.). When the staged output is DaCe's reserved
return array ``__return``, that produced ``__return_tile_out`` -- and once
``simplify()`` hoisted the transient to the top level, ``CompiledSDFG``
died on ``assert not any(aname.startswith('__return_') ...)``.
"""
import re

import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.utils.name_schemes import sanitize_transient_name_hint
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

#: Reserved return-array names: exactly ``__return`` or ``__return_<int>``.
_RETURN_ARRAY_RE = re.compile(r'^__return(_[0-9]+)?$')


def _assert_no_reserved_transients(sdfg: dace.SDFG) -> None:
    """Assert no (nested) array name sits in the reserved ``__return*``
    namespace other than the return arrays themselves."""
    for nsdfg in sdfg.all_sdfgs_recursive():
        for aname, desc in nsdfg.arrays.items():
            if aname.startswith('__return'):
                assert _RETURN_ARRAY_RE.match(aname), \
                    f"transient {aname!r} collides with the reserved __return* namespace"
                assert not desc.transient


def test_sanitize_transient_name_hint_unit():
    """The helper strips the reserved prefix and is a no-op otherwise."""
    assert sanitize_transient_name_hint("A", "_tile") == "A_tile"
    assert sanitize_transient_name_hint("_priv", "_tile_out") == "_priv_tile_out"
    for base in ("__return", "__return_0", "__return_17"):
        for suffix in ("_tile", "_tile_out", "_scatter_out", "_gather", "_const", ""):
            hint = sanitize_transient_name_hint(base, suffix)
            assert not hint.startswith("__return"), (base, suffix, hint)
    assert sanitize_transient_name_hint("__return", "_tile_out") == "tile_return_tile_out"


def test_vectorized_return_program_compiles_and_runs():
    """``return X + Y`` -> vectorize -> simplify -> C++ compile -> run.

    Exactly the bug-13 repro: simplify hoists the body transient minted from
    ``__return`` to the top level, where CompiledSDFG asserts on the name.
    """
    N = dace.symbol("N")

    @dace.program
    def vec_return_add(X: dace.float64[N], Y: dace.float64[N]):
        return X + Y

    sdfg = vec_return_add.to_sdfg()
    VectorizeCPUMultiDim(widths=(8, )).apply_pass(sdfg, {})
    sdfg.simplify()

    _assert_no_reserved_transients(sdfg)

    rng = np.random.default_rng(7)
    for n in (64, 37):  # aligned + non-divisible remainder
        X = rng.random(n)
        Y = rng.random(n)
        out = sdfg(X=X, Y=Y, N=n)
        np.testing.assert_allclose(out, X + Y, rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

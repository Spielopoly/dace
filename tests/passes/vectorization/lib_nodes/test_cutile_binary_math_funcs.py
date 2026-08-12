# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Binary elemental math functions through the cuTile (and CPU) tile-op path.

Covers ``np.arctan2`` / ``np.hypot`` / ``np.fmod`` -> ``TileBinop`` ops
``atan2`` / ``hypot`` / ``fmod``:

* ``atan2`` lowers to native ``ct.atan2`` (needs tileiras 13.2+ at runtime).
* ``hypot`` / ``fmod`` have no cuda.tile primitive, so they are decomposed with
  ``ct.sqrt`` / ``ct.floor`` / ``ct.ceil`` / ``ct.where``.
* ``np.fmod`` reaches the pass as ``cpp_mod(...)``; detection normalizes it to
  the ``fmod`` op (covers both the cuTile and CPU-CPP paths).
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from dace.transformation.passes.vectorization.config import VectorizeConfig

N = dace.symbol("N")


@dace.program
def _atan2(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
    out[:] = np.arctan2(a, b)


@dace.program
def _hypot(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
    out[:] = np.hypot(a, b)


@dace.program
def _fmod(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
    out[:] = np.fmod(a, b)


def _lower_cutile(prog):
    """Lower to the cuTile Python backend at tile width 32."""
    sdfg = prog.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(32, )).apply_pass(sdfg, {})
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


def _main_code(sdfg):
    build_objects = [co for co in sdfg.generate_code() if co.target_type == "cutile_build"]
    assert len(build_objects) == 1
    return build_objects[0].code


# ---- inputs: negatives (fmod sign / atan2 quadrants) + non-divisible size ----
def _inputs(n=37):
    rng = np.random.default_rng(0)
    a = rng.uniform(-5, 5, size=n).astype(np.float64)
    b = rng.uniform(-5, 5, size=n).astype(np.float64)
    b[np.abs(b) < 0.1] = 1.0  # keep the divisor away from zero
    return a, b


# ============================================================
# Codegen structure (no GPU)
# ============================================================


def test_atan2_emits_native_ct_atan2():
    assert "ct.atan2(" in _main_code(_lower_cutile(_atan2))


def test_hypot_decomposes_to_sqrt():
    code = _main_code(_lower_cutile(_hypot))
    assert "ct.sqrt(" in code
    assert "ct.hypot(" not in code  # no such cuda.tile primitive


def test_fmod_decomposes_and_drops_cpp_mod():
    code = _main_code(_lower_cutile(_fmod))
    # C-sign trunc via floor/ceil/where; the raw cpp_mod call must be gone.
    assert "ct.floor(" in code and "ct.ceil(" in code and "ct.where(" in code
    assert "cpp_mod" not in code
    assert "ct.fmod(" not in code  # no such cuda.tile primitive


# ============================================================
# CPU CPP path (no GPU): the cpp_mod -> fmod normalization must tile fmod
# ============================================================


def test_fmod_cpu_matches_numpy():
    sdfg = _fmod.to_sdfg(simplify=False)
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ))).apply_pass(sdfg, {})
    csdfg = sdfg.compile()
    a, b = _inputs()
    out = np.zeros_like(a)
    csdfg(a=a, b=b, out=out, N=a.size)
    np.testing.assert_allclose(out, np.fmod(a, b), rtol=1e-12, atol=1e-12)


# ============================================================
# End-to-end GPU runtime vs NumPy
# ============================================================


@pytest.mark.gpu
class TestRuntime:

    def _run(self, prog, npfun):
        csdfg = _lower_cutile(prog).compile()
        a, b = _inputs()
        out = np.zeros_like(a)
        csdfg(a=a, b=b, out=out, N=a.size)
        np.testing.assert_allclose(out, npfun(a, b), rtol=1e-12, atol=1e-12)

    def test_atan2(self):
        try:
            self._run(_atan2, np.arctan2)
        except Exception as e:  # native ct.atan2 needs tileiras 13.2+
            if "tileiras" in str(e).lower() or "atan2" in str(e).lower():
                pytest.skip(f"cuda.tile runtime lacks native atan2: {e}")
            raise

    def test_hypot(self):
        self._run(_hypot, np.hypot)

    def test_fmod(self):
        self._run(_fmod, np.fmod)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

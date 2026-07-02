# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for Bugs 05 / 14 -- ``GPU_Global`` transients accessed on
host in the cuTile pipeline.

Sequential solvers such as ``cholesky`` / ``trisolv`` (and accumulator kernels
like ``go_fast``) mix device tiles / device library results with host-resident
scalar control flow. ``apply_gpu_transformations()`` blindly places every
transient on ``GPU_Global`` and stamps every map ``GPU_Device``, so:

* a host-scheduled CuPy BLAS ``Dot`` tasklet writes a ``GPU_Global`` ``_result``
  scalar, and
* the sequential scalar-control maps stay ``GPU_Device``,

which produced ``InvalidSDFGEdgeError: Data container "_result" is stored as
StorageType.GPU_Global but accessed on host`` at validation, and (once that was
lifted) ``KeyError: ScheduleType.GPU_Device`` at Python-backend code generation.

The fix:

* ``_accessible`` (``dace/sdfg/validation.py``) treats ``GPU_Global`` as
  host-addressable for Python-backend SDFGs -- there it is a ``cupy`` array;
* ``GPUDeviceToCuTile`` demotes every non-tileops ``GPU_Device`` map to
  ``Sequential`` so it is emitted as a host driver loop over the ``cupy``
  operands;
* ``ExpandDotCuPy`` ravels its operands (a ``A[i, :j]`` slice arrives as a
  ``(1, j)`` view) so ``cupy.dot`` returns a scalar;
* ``sympy_function_redefinitions`` exposes the elementwise math functions the
  host scalar tasklets emit verbatim (e.g. ``sqrt``).

The structural tests guard the pipeline / validation without a GPU; the runtime
tests validate numerics on a GPU.
"""
import warnings

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile


def _gpu_device_maps(sdfg: SDFG):
    """Return every map still scheduled ``GPU_Device`` after lowering.

    :param sdfg: The lowered SDFG to inspect (NestedSDFGs included).
    :returns: List of ``MapEntry`` nodes still on ``GPU_Device``.
    """
    return [
        node for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device
    ]


def _trisolv_sdfg(strict: bool = False) -> SDFG:
    """Lower a symbolic triangular solver through the cuTile front door.

    :param strict: Forwarded to :class:`VectorizeCuTile`; ``True`` exercises
        the supported BLAS-only-under-strict configuration.
    """
    N = dace.symbol("N", dtype=dace.int64)

    @dace.program
    def trisolv(L: dace.float64[N, N], x: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]

    sdfg = trisolv.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(8, 8), strict=strict).apply_pass(sdfg, {})
    return sdfg


def _cholesky_sdfg(strict: bool = False) -> SDFG:
    """Lower a symbolic Cholesky factorization through the cuTile front door.

    Exercises the host ``np.dot`` (CuPy ``Dot``) result *and* the ``np.sqrt``
    scalar tasklet.

    :param strict: Forwarded to :class:`VectorizeCuTile`.
    """
    N = dace.symbol("N", dtype=dace.int64)

    @dace.program
    def cholesky(A: dace.float64[N, N]):
        A[0, 0] = np.sqrt(A[0, 0])
        for i in range(1, N):
            for j in range(i):
                A[i, j] -= np.dot(A[i, :j], A[j, :j])
                A[i, j] /= A[j, j]
            A[i, i] -= np.dot(A[i, :i], A[i, :i])
            A[i, i] = np.sqrt(A[i, i])

    sdfg = cholesky.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(8, 8), strict=strict).apply_pass(sdfg, {})
    return sdfg


def test_trisolv_lowers_and_validates():
    """trisolv lowers with no residual GPU_Device map and passes validation.

    Guards the Bug 05 / 14 ``GPU_Global``-accessed-on-host ``InvalidSDFGEdgeError``
    (from the CuPy ``Dot`` ``_result`` scalar) and the follow-on
    ``KeyError: ScheduleType.GPU_Device``. Pure structural check -- no GPU.
    """
    sdfg = _trisolv_sdfg()
    assert sdfg.backend == dtypes.BackendLanguage.Python
    assert not _gpu_device_maps(sdfg), "leftover GPU_Device maps must be demoted to Sequential"
    # Would raise InvalidSDFGEdgeError before the validation fix.
    sdfg.validate()


def test_cholesky_lowers_and_validates():
    """cholesky lowers, validates, and leaves no GPU_Device map. No GPU."""
    sdfg = _cholesky_sdfg()
    assert not _gpu_device_maps(sdfg)
    sdfg.validate()


@pytest.mark.parametrize("build", [_trisolv_sdfg, _cholesky_sdfg], ids=["trisolv", "cholesky"])
def test_blas_only_strict_lowers_and_validates(build):
    """``strict=True`` no longer hard-fails on the supported BLAS-only
    configuration (zero tileops anchors, all reductions became BLAS ``Dot``).
    Only the informational BLAS-only warning fires. No GPU."""
    with pytest.warns(UserWarning, match="BLAS-only configuration"):
        sdfg = build(strict=True)
    assert sdfg.backend == dtypes.BackendLanguage.Python
    assert not _gpu_device_maps(sdfg)
    sdfg.validate()


@pytest.mark.parametrize("build", [_trisolv_sdfg, _cholesky_sdfg], ids=["trisolv", "cholesky"])
def test_blas_only_no_demotion_warning(build):
    """The demoted scalar-control maps of trisolv/cholesky are all provably
    tiny (volume 1), so the size-gated demotion diagnostic must stay silent."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build()
    assert not [w for w in caught if "per-element host loop" in str(w.message)]


def test_return_array_name_excludes_tile_transients():
    """The Python-backend return marshaler only treats ``__return`` /
    ``__return_<int>`` as return values, not ``__return_tile*`` transients."""
    from dace.codegen.py.compiled_sdfg import _is_return_array_name
    assert _is_return_array_name("__return")
    assert _is_return_array_name("__return_0")
    assert _is_return_array_name("__return_12")
    assert not _is_return_array_name("__return_tile")
    assert not _is_return_array_name("__return_tile_out")
    assert not _is_return_array_name("__returnish")


@pytest.mark.gpu
@pytest.mark.parametrize("n", [16, 20])
def test_trisolv_cutile_runtime(n: int):
    """Compile+run trisolv on GPU and validate numerics (divisible and not).

    The CuPy ``Dot`` ``_result`` lives on ``GPU_Global`` and is written by a
    host tasklet; the sequential ``i`` loop is host driver code over the cupy
    arrays. Covers a non-divisible boundary (``n=20`` with width 8).
    """
    sdfg = _trisolv_sdfg()
    csdfg = sdfg.compile()

    rng = np.random.default_rng(0)
    # Well-conditioned lower-triangular system.
    L = np.tril(rng.random((n, n))) + n * np.eye(n)
    b = rng.random(n)
    x = np.zeros(n)

    ref = np.zeros(n)
    for i in range(n):
        ref[i] = (b[i] - L[i, :i] @ ref[:i]) / L[i, i]

    csdfg(L=L.copy(), x=x, b=b.copy(), N=n)
    assert np.allclose(x, ref, atol=1e-9)


@pytest.mark.gpu
def test_trisolv_strict_cutile_runtime():
    """GPU e2e for the strict BLAS-only path: ``VectorizeCuTile(strict=True)``
    lowers trisolv, which compiles, runs, and matches NumPy."""
    n = 20
    sdfg = _trisolv_sdfg(strict=True)
    csdfg = sdfg.compile()

    rng = np.random.default_rng(2)
    L = np.tril(rng.random((n, n))) + n * np.eye(n)
    b = rng.random(n)
    x = np.zeros(n)

    ref = np.zeros(n)
    for i in range(n):
        ref[i] = (b[i] - L[i, :i] @ ref[:i]) / L[i, i]

    csdfg(L=L.copy(), x=x, b=b.copy(), N=n)
    assert np.allclose(x, ref, atol=1e-9)


@pytest.mark.gpu
def test_cholesky_strict_cutile_runtime():
    """GPU e2e for the strict BLAS-only path on cholesky vs NumPy."""
    n = 16
    sdfg = _cholesky_sdfg(strict=True)
    csdfg = sdfg.compile()

    rng = np.random.default_rng(3)
    M = rng.random((n, n))
    A = M @ M.T + n * np.eye(n)

    ref = A.copy()
    ref[0, 0] = np.sqrt(ref[0, 0])
    for i in range(1, n):
        for j in range(i):
            ref[i, j] -= np.dot(ref[i, :j], ref[j, :j])
            ref[i, j] /= ref[j, j]
        ref[i, i] -= np.dot(ref[i, :i], ref[i, :i])
        ref[i, i] = np.sqrt(ref[i, i])

    out = A.copy()
    csdfg(A=out, N=n)
    assert np.allclose(np.tril(np.asarray(out)), np.tril(ref), atol=1e-8)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [16, 20])
def test_cholesky_cutile_runtime(n: int):
    """Compile+run cholesky on GPU and validate numerics (divisible and not).

    Exercises the ``GPU_Global`` CuPy ``Dot`` result *and* the host ``np.sqrt``
    scalar tasklet (``sympy_function_redefinitions`` must expose ``sqrt``).
    """
    sdfg = _cholesky_sdfg()
    csdfg = sdfg.compile()

    rng = np.random.default_rng(1)
    M = rng.random((n, n))
    A = M @ M.T + n * np.eye(n)  # symmetric positive-definite

    ref = A.copy()
    ref[0, 0] = np.sqrt(ref[0, 0])
    for i in range(1, n):
        for j in range(i):
            ref[i, j] -= np.dot(ref[i, :j], ref[j, :j])
            ref[i, j] /= ref[j, j]
        ref[i, i] -= np.dot(ref[i, :i], ref[i, :i])
        ref[i, i] = np.sqrt(ref[i, i])

    out = A.copy()
    csdfg(A=out, N=n)
    # Compare the factor (lower triangle incl. diagonal), which is what the
    # in-place kernel overwrites.
    assert np.allclose(np.tril(np.asarray(out)), np.tril(ref), atol=1e-8)


if __name__ == "__main__":
    test_trisolv_lowers_and_validates()
    test_cholesky_lowers_and_validates()
    test_return_array_name_excludes_tile_transients()
    test_trisolv_cutile_runtime(16)
    test_trisolv_cutile_runtime(20)
    test_cholesky_cutile_runtime(16)
    test_cholesky_cutile_runtime(20)

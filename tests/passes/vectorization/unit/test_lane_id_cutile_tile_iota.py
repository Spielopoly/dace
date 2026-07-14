# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonical-TileIota lane-id materialisation + cuTile f64 constant handling.

``materialise_lane_id_index_tile`` mints the per-lane index tile shared by
the lane-id symbol path and the gather-index path. It ALWAYS places a
``TileIota`` lib node whose ``expr`` is canonical Python source; each
expansion renders it per target ('pure' re-renders to C++ with a
``constexpr``-bounded ``DACE_UNROLL`` lane loop, 'cutile' splices Python).

``_cutile_f64_const_tile`` recovers f64 precision for float constants in
cuTile binops: cuda.tile (1.5.0) lowers host scalars in scalar-tile ops
through float32, so a non-f32-exact constant like ``0.33333`` must be
materialised as a tile built from two f32-exact terms.
"""
import ast

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.tileops.nodes import TileIota
from dace.transformation.passes.vectorization.utils.tasklets import materialise_lane_id_index_tile


def _fresh_state():
    sdfg = dace.SDFG(f"lane_id_unit_{np.random.randint(1 << 30)}")
    return sdfg.add_state()


def _tasklets(state):
    return [n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet)]


def _iotas(state):
    return [n for n in state.nodes() if isinstance(n, TileIota)]


class TestLaneIdCanonicalTileIota:

    def test_always_emits_tile_iota(self):
        """The default (CPU) path places a TileIota lib node, never a raw tasklet."""
        state = _fresh_state()
        an = materialise_lane_id_index_tile(state, "ii", ("ii", ), (8, ))
        assert not _tasklets(state)
        (iota, ) = _iotas(state)
        assert list(iota.widths) == [8]
        assert {"pure", "cutile"} <= set(iota.implementations)
        # Lane var substituted: per-lane value is ii + __l0.
        assert "__l0" in iota.expr and "ii" in iota.expr
        desc = state.sdfg.arrays[an.data]
        assert desc.dtype == dace.int64
        assert tuple(desc.shape) == (8, )
        # The consumer contract: the returned AccessNode is the iota's output.
        (edge, ) = state.out_edges(iota)
        assert edge.src_conn == "_dst" and edge.dst is an

    def test_expr_is_canonical_python(self):
        """A non-affine expr is stored as valid Python with the lane var
        substituted (``ii**2`` keeps its Python spelling)."""
        state = _fresh_state()
        materialise_lane_id_index_tile(state, "ii**2", ("ii", ), (8, ))
        (iota, ) = _iotas(state)
        ast.parse(iota.expr, mode="eval")  # must be valid Python
        assert "__l0" in iota.expr

    def test_k2_expr_substitutes_both_lane_vars(self):
        state = _fresh_state()
        materialise_lane_id_index_tile(state, "2 * ii + jj", ("ii", "jj"), (4, 8))
        (iota, ) = _iotas(state)
        assert list(iota.widths) == [4, 8]
        assert "__l0" in iota.expr and "__l1" in iota.expr

    def test_pure_expansion_renders_cpp_with_unroll_and_cast(self):
        """The 'pure' expansion re-renders the Python expr to C++: unrolled
        constexpr-bounded lane loop, dtype cast, no Python ``**``."""
        state = _fresh_state()
        materialise_lane_id_index_tile(state, "ii**2", ("ii", ), (8, ))
        (iota, ) = _iotas(state)
        iota.expand(state, "pure")
        (tasklet, ) = _tasklets(state)
        code = tasklet.code.as_string
        assert tasklet.language == dtypes.Language.CPP
        assert "DACE_UNROLL" in code
        assert "constexpr std::size_t __W0 = 8;" in code
        assert f"({dace.int64.ctype})(" in code
        assert "**" not in code  # Python power rendered as C++ multiplication

    def test_pure_expansion_keeps_py_mod_call(self):
        """``py_mod`` survives the C++ re-render as the runtime helper call."""
        state = _fresh_state()
        materialise_lane_id_index_tile(state, "py_mod(ii, 4)", ("ii", ), (8, ))
        (iota, ) = _iotas(state)
        iota.expand(state, "pure")
        (tasklet, ) = _tasklets(state)
        assert "py_mod" in tasklet.code.as_string

    def test_pure_expansion_k2_unrolls_both_dims(self):
        state = _fresh_state()
        materialise_lane_id_index_tile(state, "2 * ii + jj", ("ii", "jj"), (4, 8))
        (iota, ) = _iotas(state)
        iota.expand(state, "pure")
        (tasklet, ) = _tasklets(state)
        code = tasklet.code.as_string
        assert "constexpr std::size_t __W0 = 4;" in code
        assert "constexpr std::size_t __W1 = 8;" in code
        assert code.count("DACE_UNROLL") == 2

    def test_pure_expansion_w1_collapse(self):
        """All-ones widths: the Register (1,) array collapses to a scalar, so
        the body must not index ``_dst``."""
        state = _fresh_state()
        materialise_lane_id_index_tile(state, "ii", ("ii", ), (1, ))
        (iota, ) = _iotas(state)
        iota.expand(state, "pure")
        (tasklet, ) = _tasklets(state)
        code = tasklet.code.as_string
        assert "_dst[" not in code
        assert "_dst = " in code and "for" not in code

    def test_pure_expansion_unparseable_expr_splices_verbatim(self):
        """A hand-built node with a non-symbolic expr keeps the historical
        verbatim splice (with a warning) instead of crashing."""
        sdfg = dace.SDFG("iota_verbatim")
        state = sdfg.add_state()
        sdfg.add_array("t", (8, ), dace.int64, transient=True, storage=dtypes.StorageType.Register)
        iota = TileIota("fill", widths=(8, ), expr="do_gather(__l0, (")
        state.add_edge(iota, "_dst", state.add_access("t"), None, dace.Memlet("t[0:8]"))
        with pytest.warns(UserWarning, match="not dace.symbolic-parseable"):
            iota.expand(state, "pure")
        (tasklet, ) = (n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet))
        assert "do_gather(__l0, (" in tasklet.code.as_string

    def test_cutile_iota_expands_to_python(self):
        """The placed node's 'cutile' expansion is a Python tasklet using
        ct.arange over the declared int64 dtype."""
        state = _fresh_state()
        an = materialise_lane_id_index_tile(state, "ii", ("ii", ), (8, ))
        (iota, ) = _iotas(state)
        iota.expand(state, "cutile")
        (tasklet, ) = _tasklets(state)
        assert tasklet.language == dtypes.Language.Python
        assert "ct.arange(8, dtype=ct.int64)" in tasklet.code.as_string
        assert an.data in {e.data.data for e in state.in_edges(an)}

    def test_pure_e2e_gather_numerics(self):
        """End-to-end CPU check: a cyclic gather index (``py_mod``) built via
        the TileIota path compiles and matches the reference, non-divisible size."""
        from tests.passes.vectorization.helpers.harness import S, X, run_vectorization_test

        @dace.program
        def cyclic(out: dace.float64[X], a: dace.float64[S]):
            for i, in dace.map[0:X:1]:
                out[i] = a[i % S]

        xv, sv = 20, 4  # xv not a multiple of 8 -> remainder tile
        run_vectorization_test(
            dace_func=cyclic,
            arrays={
                "out": np.zeros(xv),
                "a": np.random.random(sv)
            },
            params={
                "X": xv,
                "S": sv
            },
            vector_width=8,
            sdfg_name="lane_id_iota_cyclic_e2e",
        )


class TestCutileF64ConstTile:

    def _helper(self):
        from dace.libraries.tileops.nodes.tile_binop import _cutile_f64_const_tile
        return _cutile_f64_const_tile

    def test_non_f32_exact_constant_splits(self):
        expr = self._helper()("0.33333", (8, ))
        assert expr is not None
        assert "ct.full((8,)," in expr.replace(" ", "")
        assert "dtype=ct.float64" in expr
        # The two emitted terms recombine to the exact f64 constant, and the
        # leading term is exactly representable in f32.
        import re
        hi, lo = (float(t) for t in re.findall(r"-?\d+\.\d+(?:e-?\d+)?", expr))
        assert hi + lo == 0.33333
        assert float(np.float32(hi)) == hi

    def test_parenthesized_symstr_form(self):
        assert self._helper()("(0.33333)", (8, )) is not None

    def test_f32_exact_constant_untouched(self):
        assert self._helper()("2.0", (8, )) is None
        assert self._helper()("0.5", (8, )) is None
        assert self._helper()("0", (8, )) is None

    def test_symbolic_expression_untouched(self):
        assert self._helper()("N + 1", (8, )) is None
        assert self._helper()("beta", (8, )) is None

    def test_k2_shape(self):
        expr = self._helper()("0.33333", (4, 8))
        assert expr is not None and "(4, 8,)" in expr

    def test_binop_expansion_uses_const_tile_for_f64(self):
        """End-to-end through the lib-node expansion: an f64 output binop with
        a non-f32-exact Symbol operand emits the two-term tile constant."""
        from dace.libraries.tileops.nodes import TileBinop
        sdfg = dace.SDFG("const_tile_binop")
        state = sdfg.add_state()
        sdfg.add_array("b", (8, ), dace.float64, transient=True, storage=dtypes.StorageType.Register)
        sdfg.add_array("c", (8, ), dace.float64, transient=True, storage=dtypes.StorageType.Register)
        node = TileBinop("mult", widths=(8, ), op="*", kind_a="Symbol", kind_b="Tile", expr_a="0.33333")
        state.add_edge(state.add_access("b"), None, node, "_b", dace.Memlet("b[0:8]"))
        state.add_edge(node, "_c", state.add_access("c"), None, dace.Memlet("c[0:8]"))
        node.expand(state, "cutile")
        (tasklet, ) = (n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet))
        body = tasklet.code.as_string
        assert "ct.full" in body and "dtype=ct.float64" in body

    def test_binop_expansion_leaves_exact_constant_inline(self):
        from dace.libraries.tileops.nodes import TileBinop
        sdfg = dace.SDFG("const_inline_binop")
        state = sdfg.add_state()
        sdfg.add_array("b", (8, ), dace.float64, transient=True, storage=dtypes.StorageType.Register)
        sdfg.add_array("c", (8, ), dace.float64, transient=True, storage=dtypes.StorageType.Register)
        node = TileBinop("mult", widths=(8, ), op="*", kind_a="Symbol", kind_b="Tile", expr_a="2.0")
        state.add_edge(state.add_access("b"), None, node, "_b", dace.Memlet("b[0:8]"))
        state.add_edge(node, "_c", state.add_access("c"), None, dace.Memlet("c[0:8]"))
        node.expand(state, "cutile")
        (tasklet, ) = (n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet))
        assert "ct.full" not in tasklet.code.as_string


if __name__ == "__main__":
    t = TestLaneIdCanonicalTileIota()
    t.test_always_emits_tile_iota()
    t.test_expr_is_canonical_python()
    t.test_k2_expr_substitutes_both_lane_vars()
    t.test_pure_expansion_renders_cpp_with_unroll_and_cast()
    t.test_pure_expansion_keeps_py_mod_call()
    t.test_pure_expansion_k2_unrolls_both_dims()
    t.test_pure_expansion_w1_collapse()
    t.test_pure_expansion_unparseable_expr_splices_verbatim()
    t.test_cutile_iota_expands_to_python()
    t.test_pure_e2e_gather_numerics()
    c = TestCutileF64ConstTile()
    c.test_non_f32_exact_constant_splits()
    c.test_parenthesized_symstr_form()
    c.test_f32_exact_constant_untouched()
    c.test_symbolic_expression_untouched()
    c.test_k2_shape()
    c.test_binop_expansion_uses_const_tile_for_f64()
    c.test_binop_expansion_leaves_exact_constant_inline()

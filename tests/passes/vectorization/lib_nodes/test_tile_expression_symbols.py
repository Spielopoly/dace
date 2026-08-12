# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Symbol discovery and replacement for string-valued tile expressions."""
import pytest

import dace
from dace.libraries.tileops import TileIota, TileLoad, TileStore
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols


def test_tile_iota_expression_symbols_and_replacement():
    """Only runtime names in an iota expression are SDFG symbols."""
    sdfg = dace.SDFG("replace_tile_iota_expression")
    state = sdfg.add_state("main")
    node = TileIota(name="iota", widths=(8, ), expr="offset + _idx[__l0]", extra_inputs=("_idx", ))
    state.add_node(node)

    sdfg.add_symbol("offset", dace.int64)
    sdfg.add_symbol("dead", dace.int64)
    RemoveUnusedSymbols().apply_pass(sdfg, {})
    assert set(sdfg.symbols) == {"offset"}

    assert node.free_symbols == {"offset"}
    sdfg.replace_dict({"offset": "base + 1"}, replace_keys=False)
    assert node.free_symbols == {"base"}
    assert "offset" not in node.expr
    assert "_idx[__l0]" in node.expr


@pytest.mark.parametrize("node_type", [TileLoad, TileStore])
def test_tile_source_expression_symbols_and_replacement(node_type):
    """Symbol-valued tile loads and stores retain and rename dependencies."""
    sdfg = dace.SDFG(f"replace_{node_type.__name__.lower()}_expression")
    state = sdfg.add_state("main")
    node = node_type(name="symbol_source", widths=(8, ), src_kind="Symbol", src_expr="scale + 6.25j")
    state.add_node(node)

    sdfg.add_symbol("scale", dace.float64)
    sdfg.add_symbol("dead", dace.int64)
    RemoveUnusedSymbols().apply_pass(sdfg, {})
    assert set(sdfg.symbols) == {"scale"}

    assert node.free_symbols == {"scale"}
    sdfg.replace_dict({"scale": "new_scale * 2"}, replace_keys=False)
    assert node.free_symbols == {"new_scale"}
    assert "new_scale" in node.src_expr
    assert "6.25j" in node.src_expr

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Truth-table tests for :func:`select_tile_implementation` (``_dispatch``).

Locks the dispatch semantics: ``K >= 2`` always resolves to ``'pure'`` for
EVERY ``target_isa`` (including ``CUTILE`` — cuTile lowering stamps
``implementation = 'cutile'`` directly and never goes through dispatch);
``K == 1`` maps ``target_isa`` through ``_ISA_TO_IMPL`` (``AUTO`` resolves
via :func:`detect_host_isa`), falling back to ``'pure'`` when the ISA is
unknown or the node does not define that implementation.
"""
from typing import Tuple

import pytest

from dace.libraries.tileops import TileBinop, TileMaskGen
from dace.libraries.tileops import _dispatch
from dace.libraries.tileops._dispatch import (
    _ISA_TO_IMPL,
    detect_host_isa,
    select_tile_implementation,
)


def _binop(widths: Tuple[int, ...], target_isa: str) -> TileBinop:
    """Build a minimal :class:`TileBinop` with the given dispatch inputs.

    :param widths: Per-dim tile widths (sets ``K = len(widths)``).
    :param target_isa: Value stamped on ``node.target_isa``.
    :returns: A standalone node (dispatch needs no SDFG context).
    """
    node = TileBinop(name="tb", widths=widths, op="+")
    node.target_isa = target_isa
    return node


@pytest.mark.parametrize("target_isa", ["AUTO", "AVX512", "AVX2", "ARM_SVE", "ARM_NEON", "SCALAR", "CUTILE"])
@pytest.mark.parametrize("widths", [(8, 4), (4, 4, 2)])
def test_k_ge_2_is_always_pure(widths: Tuple[int, ...], target_isa: str):
    """K >= 2 resolves to ``'pure'`` for every ISA — including ``CUTILE``."""
    assert select_tile_implementation(_binop(widths, target_isa)) == "pure"


@pytest.mark.parametrize(
    "target_isa, expected",
    [
        ("AVX512", "avx512"),
        ("AVX2", "avx2"),
        ("ARM_SVE", "sve"),
        ("ARM_NEON", "neon"),
        ("SCALAR", "scalar"),
        ("CUTILE", "cutile"),
    ],
)
def test_k1_maps_isa_to_node_implementation(target_isa: str, expected: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """K == 1 maps known ISAs independently of host safety validation."""
    monkeypatch.setattr(_dispatch, "host_supported_isas", lambda: frozenset(_ISA_TO_IMPL))
    assert select_tile_implementation(_binop((8, ), target_isa)) == expected


def test_k1_unknown_isa_falls_back_to_pure():
    """K == 1 with an ISA outside ``_ISA_TO_IMPL`` falls back to ``'pure'``."""
    assert select_tile_implementation(_binop((8, ), "FOO")) == "pure"


def test_k1_unstamped_node_dispatches_as_scalar():
    """A freshly constructed node (orchestrator never stamped ``target_isa``)
    carries the property default ``'SCALAR'`` and dispatches accordingly."""
    node = TileBinop(name="tb", widths=(8, ), op="+")
    assert node.target_isa == "SCALAR"
    assert select_tile_implementation(node) == "scalar"


def test_k1_auto_resolves_via_host_isa():
    """``AUTO`` resolves through :func:`detect_host_isa`; on ``TileBinop``
    (which defines every per-ISA expansion) the result is exactly the
    host ISA's implementation name."""
    expected = _ISA_TO_IMPL[detect_host_isa()]
    assert select_tile_implementation(_binop((8, ), "AUTO")) == expected


def test_k1_cutile_on_node_with_cutile_impl_resolves_cutile():
    """K == 1 + CUTILE on a node that defines ``'cutile'`` resolves to it,
    even when the node lacks every CPU-ISA expansion (``TileMaskGen``)."""
    node = TileMaskGen(name="tmg", widths=(8, ), iter_vars=("i", ), global_ubs=("N", ))
    node.target_isa = "CUTILE"
    assert select_tile_implementation(node) == "cutile"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

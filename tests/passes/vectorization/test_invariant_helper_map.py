# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.transformation.passes.vectorization.utils.map_predicates import (is_vectorizable_map,
                                                                           map_body_depends_on_tiled_params)


def test_loop_invariant_helper_map_is_refused():
    """A fixed-size GPU helper map around scalar arithmetic has no lane semantics."""
    sdfg = dace.SDFG("invariant_helper")
    sdfg.add_array("a", (1, ), dace.float64)
    sdfg.add_array("out", (1, ), dace.float64)
    state = sdfg.add_state("main")
    map_entry, map_exit = state.add_map("helper", {"lane": "0:32"})
    tasklet = state.add_tasklet("scalar_add", {"a"}, {"out"}, "out = a + 1")
    state.add_memlet_path(state.add_access("a"), map_entry, tasklet, dst_conn="a", memlet=dace.Memlet("a[0]"))
    state.add_memlet_path(tasklet, map_exit, state.add_access("out"), src_conn="out", memlet=dace.Memlet("out[0]"))

    assert map_body_depends_on_tiled_params(state, map_entry, ("lane", )) is False
    assert is_vectorizable_map(state, map_entry) is False

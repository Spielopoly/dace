# node: tile_iota — running log

**Status:** TESTING-COMPLETE (batch A, all GPU runtime + unit tests pass — awaiting human go/no-go)
**File:** `dace/dace/libraries/tileops/nodes/tile_iota.py` (260 lines)
**New runtime tests:** `tests/passes/vectorization/lib_nodes/test_tile_iota_cutile_runtime.py`

## Current state & gaps (from CURRENT_STATE.md)
- Emits the `expr` string verbatim as Python — CPP-flavored input is a silent crash trap.
- `ct.arange` hardwired int32 (no dtype generality).
- Only indirect runtime coverage (rides on other fixtures).

## Edge-case matrix to cover
- `ct.arange` is **1-D only**; multi-dim iota must be built via reshape/broadcast across axes.
- dtype generality (int32/int64/float variants) via `astype`.
- affine expressions (start/step), per-axis iota in K≥2.
- non-divisible sizes / boundary lanes; symbolic sizes.
- combination with downstream ops (iota feeding gather indices, mask gen, etc.).

## Changes made (recovered from on-disk diff 2026-06-24)
- **dtype-aware `ct.arange`**: new `_resolve_dst_cutile_dtype(node, state, sdfg)` reads the wired
  `_dst` descriptor and emits `ct.arange(..., dtype=<resolved>)` (falls back to `ct.int32` for
  bare expansion with no live SDFG). Fixes the "ct.arange hardwired int32" gap. Applies to both
  K==1 and K>=2 (broadcast) paths.
- **C++-expr guard**: new `_validate_cutile_expr(expr)` raises on unambiguously-C++ constructs
  (`std::`, `->`, trailing `;`, `sizeof`) before emitting `expr` verbatim as Python — converts the
  silent crash trap into a clear error.
- **Shared dtype map extraction**: the DaCe→cuTile dtype table moved to new module
  `dace/libraries/tileops/_cutile_dtypes.py` (`_DACE_TO_CUTILE_DTYPE` + `dace_dtype_to_cutile_str`),
  now imported by tile_iota and tile_reduce. The map is the full set (all float/int widths,
  signed+unsigned, bool→`ct.bool_`).

## Coordinator fixups (stale tests from mid-refactor)
- `tests/passes/vectorization/unit/test_tile_iota_cutile_robustness.py` had stale imports/asserts
  written against the *pre-refactor* dtype map. Fixed by the coordinator:
  - import `_DACE_TO_CUTILE_DTYPE` from `_cutile_dtypes` (not tile_iota).
  - `test_dtype_mapping_covers_all_expected_types` → full 12-key set.
  - `test_dtype_mapping_values_use_ct_prefix` → special-case `bool`→`ct.bool_`.
  - replaced `test_resolve_dtype_unmapped_falls_back_to_int32` (uint8 is now *supported* → `ct.uint8`)
    with `test_resolve_dtype_uint8` + `test_resolve_dtype_falls_back_to_int32_without_descriptor`.

## Tests (commands + results)
- New `tests/passes/vectorization/lib_nodes/test_tile_iota_cutile_runtime.py` (22 KB) → **PASS**.
- New `tests/passes/vectorization/unit/test_tile_iota_cutile_robustness.py` → **43 passed**
  (after coordinator fixups above).

## Decisions needed / open issues
- None outstanding. `expr` grammar = Python; CPP constructs now rejected at expansion time.

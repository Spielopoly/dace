# node: tile_reduce — running log

**Status:** TESTING-COMPLETE (batch A, all GPU runtime tests pass — awaiting human go/no-go)
**File:** `dace/dace/libraries/tileops/nodes/tile_reduce.py` (385 lines)
**New runtime tests:** `tests/passes/vectorization/lib_nodes/test_tile_reduce_cutile_runtime.py`

## Current state & gaps (from CURRENT_STATE.md)
- Dead/misleading docstring promising a `ct.where`-absent fallback never implemented.
- Raw `axis` passthrough (not validated/normalized).
- Full-reduction → scalar seam unverified on device.
- Runtime was deliberately deferred — was blocked on a `MarkTileDims` bug (verify if still blocking).

## Edge-case matrix to cover
- ops: sum/prod/max/min (+ argmax/argmin if feasible) — use built-ins w/ axis tuple + keepdims.
- single-axis, multi-axis, and full reduction → scalar.
- non-divisible sizes (remainder lanes must use the op's identity, NOT garbage — esp. prod=1, max=-inf).
- multi-dim K≥2; symbolic sizes; mixed dtypes; numeric parity (rounding_mode/flush_to_zero if needed).
- arbitrary WCR via `ct.reduce`/`ct.scan` custom func/identity (stretch goal).

## Changes made (recovered from on-disk diff 2026-06-24)
- **Correct identity literals via numpy introspection**: rewrote `_identity_literal_cutile(op, dtype)`
  to take a `dace.dtypes.typeclass` and use `np.issubdtype` + `np.iinfo(nptype).max/min` for
  per-width signed/unsigned int extremes, `±inf` for floats, True/False for bool. Replaces the
  fragile string-parsing bitwidth helpers (`_is_cutile_*_type`, `_cutile_integer_bitwidth`) which
  were deleted. Fixes wrong min/max identities on non-int32 widths.
- **`ct.astype` positional args**: identities now emit `ct.astype(0, ct.float64)` (positional-only),
  not the previously-wrong `dtype=` keyword form.
- **Masked path genuinely consumes `_mask`** via `ct.where(_mask, _src, IDENTITY)` before the
  built-in reduction (`ct.sum/prod/min/max(..., axis=node.axis)`). Removed the dead/misleading
  "ct.where-absent fallback" docstring.
- Uses shared `dace_dtype_to_cutile_str` from `_cutile_dtypes.py` (backward-compat alias
  `_dace_dtype_to_cutile_str` kept for importers).

## Shared-file change (cutile_target.py — coordinate w/ other batches)
- `_needs_gather_scatter` (cutile_target.py ~L558): added an early `return True` when
  `len(tile_shape) < len(entry.map.range)` (tile has fewer dims than the map, e.g. after axis
  reduction). This is what **unblocked reduce runtime** — the old `MarkTileDims`-era blocker.
  The direct `ct.load`/`ct.store` path would otherwise build a rank-mismatched index. Minimal,
  scoped; flagged here so batch B/C don't clobber it.

## Tests (commands + results)
- New file `tests/passes/vectorization/lib_nodes/test_tile_reduce_cutile_runtime.py` (33 KB).
- `pytest .../test_tile_reduce_cutile_runtime.py` → **PASS** (part of 164-passed batch-A run).
  Covers K=1/K=2 sum/prod/min/max, axis0/full, masked, non-divisible, expansion-structure checks.

## Decisions needed / open issues
- The `MarkTileDims` blocker was resolved at the codegen layer (dim-mismatch → gather/scatter),
  not in `MarkTileDims` itself. Not logged in CORE_BUGFIXES.md (it's a cuTile-codegen behavior,
  not a core-DaCe transform bug). Revisit if a cleaner fix in the pass is preferred.

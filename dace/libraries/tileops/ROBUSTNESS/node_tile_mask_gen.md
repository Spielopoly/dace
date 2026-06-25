# node: tile_mask_gen — running log

**Status:** TESTING-COMPLETE (batch B, GPU runtime + regression pass — coordinator-verified 2026-06-24)
**File:** `dace/dace/libraries/tileops/nodes/tile_mask_gen.py`
**New runtime tests:** `tests/passes/vectorization/lib_nodes/test_tile_mask_gen_cutile_runtime.py`

## Current state & gaps (from CURRENT_STATE.md)
Props: `widths`, `iter_vars` (per-dim map iter-var names), `global_ubs` (per-dim exclusive
upper-bound exprs). One bool output `_o`. cuTile path: PID offset via `cutile_grid_dim_offset`
(assumes K tiled dims are trailing grid axes); per dim `__mask{k} = arange(W_k)+__pid{k}*W_k < ub`;
K=1 → `_o=__mask0`; K≥2 → `&`-conjunction of broadcast masks.
Gaps:
- **`global_ubs` inlined verbatim** as Python — CPP-flavored ub = invalid Python (same grammar
  trap as TileIota `expr`). Apply the same `_validate_cutile_expr`-style guard.
- **Only the `<` (upper-bound OOB-tail) form** — no lower-bound / interior / arbitrary predicate.
- **`cutile_grid_dim_offset` positional assumption** — correct only when the K tiled dims are the
  innermost K grid axes; non-trailing dims bind the wrong `ct.bid`.
- **`ct.arange` int32** — overflow caveat (mirror tile_load `_needs_int64`).
- **`iter_vars` ignored by the cuTile path** (recomputes from `__pid{k}*W_k`) — divergence from the
  pure path if the map binding is not exactly `__pid*W`.

## Edge-case matrix to cover
- K=1,2,3 boundary masks; symbolic upper bounds; non-divisible sizes (tail lanes).
- all-true (size multiple of width) / all-false / partial masks.
- mask feeding downstream masked load/store/reduce/binop (combination robustness).
- non-trailing tiled dims (if reachable) — verify correct grid-axis binding or document the limit.
- CPP-flavored `global_ubs` → rejected with a clear error, not silent crash.

## Plan (orchestrator, approved by coordinator 2026-06-24)
1. **Hoist `_validate_cutile_expr` + `_CPP_PATTERN`** from tile_iota.py to **`_pure_codegen.py`**
   (decision approved: option C — that module already hosts cuTile helpers `cutile_grid_dim_offset`,
   `cutile_tile_dim_bids` and is universally imported). SHARED-FILE edits: `_pure_codegen.py` (+~15),
   `tile_iota.py` (import source change, remove local copy ~15). **Serialize w/ other batches.**
2. Add to `ExpandTileMaskGenCutile`: validate each `global_ubs[k]` via the guard;
   `_mask_needs_int64(global_ubs)` (numeric > 2^31-1 OR symbolic → `ct.int64`); trailing-K-param
   vs `iter_vars` match check (clear error on mismatch); docstrings for Gaps 2 & 5.
3. Gaps 2 (only `<`-OOB form) & 5 (`iter_vars` unused in cuTile path) judged **correct-by-design**
   for the remainder-tail use case → documented, no code change.
4. ~12 GPU runtime tests incl. combination tests (mask → masked load/store/reduce). APPROVED.

## Changes made (orchestrator, coordinator-verified 2026-06-24)
`tile_mask_gen.py` (+86/-7): `_mask_needs_int64(global_ubs)` (symbolic OR >2^31-1 → int64);
three guards in `ExpandTileMaskGenCutile.expansion()` — (1) `validate_cutile_expr(ub)` per
`global_ubs`, (2) trailing-K grid-axis binding check vs `iter_vars` (ValueError on mismatch),
(3) dynamic `ct.arange` dtype; docstrings for the by-design `<`-only form + unused `iter_vars`.

### SHARED-FILE edits (serialized — confirmed non-overlapping with binop/store)
- `_pure_codegen.py` (+27, additive): `validate_cutile_expr(expr)` + `_CPP_PATTERN` + `import re`.
- `tile_iota.py`: removed local `_validate_cutile_expr`/`_CPP_PATTERN`, now imports
  `validate_cutile_expr` from `_pure_codegen`; call site updated.
- `test_tile_iota_cutile_robustness.py`, `test_tile_iota_cutile_runtime.py`: import from `_pure_codegen`.
- `test_cutile_expansions.py`: one assertion `ct.int32`→`ct.int64` in
  `test_tile_mask_gen_cutile_1d_uses_arange_and_bid` (symbolic UBs now conservatively int64).

## Tests (commands + results)
- NEW `test_tile_mask_gen_cutile_runtime.py` (903 lines, 51 tests: 27 GPU runtime + 24 validation).
- Coordinator re-ran mask_gen runtime+pure + iota robustness+runtime → **122 passed (15s)**, no
  iota regression from the shared hoist.
- Session report: `TileIR/ai_session_reports/2026-06-24/tile_mask_gen_cutile_robustness.md`.

## Decisions needed / open issues
- RESOLVED. `_validate_cutile_expr` hoisted to `_pure_codegen.validate_cutile_expr` (option C).
- Minor deferred: `_INT32_MAX` duplicated (also in tile_load) — fold into coordinator's post-batch-B
  `_pure_codegen` consolidation of `_needs_int64`.

## Tests (commands + results)
(none yet)

## Decisions needed / open issues
- Reuse the iota C++-expr guard? (`_validate_cutile_expr` lives in tile_iota.py — consider
  hoisting to a shared module to avoid duplication; coordinate as a shared-file change.)

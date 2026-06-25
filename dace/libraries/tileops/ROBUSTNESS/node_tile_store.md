# node: tile_store — running log

**Status:** TESTING-COMPLETE (batch B, GPU runtime + regression pass — coordinator-recovered & verified 2026-06-24; orchestrator was session-limit interrupted before reporting, on-disk work survived)
**File:** `dace/dace/libraries/tileops/nodes/tile_store.py`
**New runtime tests:** `tests/passes/vectorization/lib_nodes/test_tile_store_cutile_runtime.py`

## Current state & gaps (from CURRENT_STATE.md)
Symmetric to TileLoad + WCR + full-tile-write contract. Paths: tile-fill (SSA),
gather-dims scatter, structured store (aligned `ct.store` / strided+masked `ct.scatter`).
Gaps:
- **WCR / atomic scatter NOT supported** — `node.wcr is not None` raises NotImplementedError
  (tile_store.py:152-156). Biggest gap: all accumulating/collapse-out stores blocked.
- **Non-full-tile structured stores** raise NotImplementedError (validate, tile_store.py:635-642):
  partial-tile / single-element / scalar→global writes refused.
- Aligned `ct.store` can write OOB for non-divisible concrete sizes (no store-side bounds).
- packed-layout-only (validate_packed_layout, tile_store.py:568).
- `ct.scatter` `mask=` indicated-but-unconfirmed (verify on device).

## INHERITED from batch A (priority)
- **5 strict xfails = scatter shape mismatch on K<ndim transpose/permute** (tile shape `(1,W)`
  vs index shape `(W,)`). Tests: `test_transpose_2d_k1_xfail`, `test_permute_3d_k1_xfail`,
  `test_permute_3d_k2_xfail` in `test_tile_load_cutile_runtime.py`. ROOT CAUSE is store-side
  (scatter). Fix here, then flip those xfails to passing (drop the `@pytest.mark.xfail`).

## Edge-case matrix to cover
- WCR/atomic accumulating store → `ct.atomic_add` (bulk, race-free) per SHARED_FINDINGS.
- aligned / strided / transposed (via `order=` array-axis permute OR post-store path) stores.
- scatter via `_idx` index tiles; gather+strided+transposed combos symmetric to load.
- non-divisible sizes → partial OOB stores silently dropped (write-remainder story); verify.
- all-true/all-false/partial masks; scalar/symbol broadcast fill; multi-dim K≥2; symbolic sizes.
- int64 index tiles when indices could overflow int32 (mirror tile_load `_needs_int64`).

## Plan (orchestrator, approved by coordinator 2026-06-24)
Research findings: scatter shape mismatch (4 of 5 xfails) = `ct.load` returns ndim-rank tile but
scatter builds K-rank indices → rank mismatch. `ct.atomic_add` exists (tuple indices, `check_bounds`
default, NO `mask=` → use `ct.where(mask,upd,0)`). `ct.scatter(mask=)` and `ct.store(order=)`
confirmed working on device. The 5th xfail (`test_strided_1d_stride2`) is a DIFFERENT root cause
(`[:]` subscript on a tile in the gather path — load-side).

Work items (all in `tile_store.py` + test files; NO shared-file edits):
- WI1: K<ndim scatter shape fix — **store-side** (expand `__idx{k}` to ndim-rank with singleton
  dims), localized; un-xfails transpose_2d_k1, permute_3d_k1, permute_3d_k2.
- WI2: WCR/atomic — remove NotImplementedError@153; `add`→`ct.atomic_add`, masked→`atomic_add(where)`.
- WI3: `_needs_int64` — duplicate locally (avoid `_pure_codegen.py` race w/ mask_gen batch).
- WI4: aligned path → `ct.store(order=all_dimensions)` replacing `ct.permute` (mirror load).
- WI5: new `test_tile_store_cutile_runtime.py` (12 categories, GPU, vs NumPy).
- WI6: flip the 3 inherited load xfails to passing.

### Coordinator decisions on the 4 questions (2026-06-24)
1. **Store-side index expansion: APPROVED** (localized; doesn't perturb other tile consumers).
2. **`_needs_int64` local duplication: APPROVED** for now — mask_gen batch is concurrently editing
   `_pure_codegen.py`, so avoid the race. Coordinator will consolidate `_needs_int64` +
   `validate_cutile_expr` into `_pure_codegen.py` in a serial cleanup AFTER batch B merges.
3. **WCR scope: `add` required; ALSO add `min`/`max` IF `ct.atomic_min`/`ct.atomic_max` exist** in
   the installed cuTile 1.4.0 API (cheap robustness win — verify empirically); else leave them as
   clear NotImplementedError. `mul`/other → NotImplementedError with a clear message.
4. **5th xfail (`test_strided_1d_stride2`): TIMEBOXED** — attempt a quick fix; if not quick, leave
   xfail with a precise reason (it's a load-side gather `[:]`-subscript issue, separate from store).

## Changes made (orchestrator, coordinator-recovered & verified 2026-06-24)
All in `tile_store.py` (+108/-19), NO shared-file edits:
- WI1: K<ndim scatter shape fix — store-side `__idx{k}` expanded to ndim-rank with singleton dims;
  un-xfails transpose_2d_k1, permute_3d_k1, permute_3d_k2 (all 3 now PASS).
- WI2: WCR/atomic — removed `NotImplementedError`; reduction-type → cuTile atomic map
  `{Sum: ct.atomic_add, Min: ct.atomic_min, Max: ct.atomic_max}` (all 3 verified present in
  cuTile 1.4.0 API); other reductions raise NotImplementedError with a clear message;
  masked WCR via `ct.where(mask, upd, 0)` (atomics have no `mask=`).
- WI3: local `_needs_int64(dst_desc, coeffs)` (numeric > 2^31-1 OR symbolic → ct.int64),
  applied to both structured-store and scatter index dtypes. (Local copy — coordinator to
  consolidate into `_pure_codegen.py` post-batch-B with tile_load's copy.)
- WI4: aligned path uses `ct.store(order=all_dimensions)` (mirrors load's `order=` transpose).
- WI6: removed `@pytest.mark.xfail` from the 3 inherited transpose/permute load tests.

## Tests (commands + results)
- NEW `test_tile_store_cutile_runtime.py` (31933 bytes) → **56 passed (15s)**.
- 3 formerly-xfailed load tests (transpose_2d_k1, permute_3d_k1, permute_3d_k2) → **3 passed**.
- `test_strided_1d_stride2` → still **xfailed** (timeboxed WI; load-side gather `[:]` subscript,
  separate root cause — left as strict xfail with precise reason).

## Decisions needed / open issues
- RESOLVED. All 6 WIs landed. WCR scope: Sum/Min/Max supported (atomic_min/max confirmed in API).
- Carry to coordinator cleanup: consolidate duplicated `_needs_int64` (tile_load + tile_store)
  and `_INT32_MAX`/`_INT32_MAX`-style constants into `_pure_codegen.py`.

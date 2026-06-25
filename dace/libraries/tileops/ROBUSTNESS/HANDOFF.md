# HAND-OFF — cuTile Expansion Robustness Effort

> **New/continuing agent: read this first, then `PROGRESS.md`, then `README.md`,
> `SHARED_FINDINGS.md`, and the relevant `node_*.md`.** This file is the
> point-in-time snapshot of exactly where the effort stopped (2026-06-25).

## Mission (unchanged)

Make the **cuTile expansions** of every tileops library node in
`dace/dace/libraries/tileops/nodes/` "as robust as humanly possible" — able to
handle arbitrary programs and **arbitrary combinations** of operations.

- **Priority 1 (hard requirement): correctness**, verified by **GPU runtime
  tests** (`@pytest.mark.gpu`) end-to-end: `@dace.program` → `canonicalize()` →
  `VectorizeCuTile` → `sdfg.backend = dace.dtypes.BackendLanguage.Python` →
  compile → run on GPU → compare vs NumPy.
- **Priority 2: speed** (only after correctness is proven).

## How the effort is run (the model the human chose)

- A top-level **coordinator** (the main Claude agent) spawns one
  **coding-orchestrator** subagent per node, talks to it as if it were the user,
  and **relays only the BIG decisions to the human**. Orchestrators' own
  `AskUserQuestion` may not reach the human — the coordinator is the bridge.
  Orchestrators lack the `Write` tool, so **the coordinator owns all `node_*.md`
  / `PROGRESS.md` / this file**.
- **Cadence: small parallel batches with go/no-go between batches.** Within a
  batch, orchestrators run autonomously to passing GPU tests; the coordinator
  summarizes and gets a human go-ahead before the next batch.
- **Batches:** A = {tile_load, tile_reduce, tile_iota}, B = {tile_store,
  tile_binop, tile_mask_gen}, **C = {tile_mma, tile_unop, tile_ite}** ← NEXT.
- **Recovery pattern (used twice, works):** orchestrators get killed by the
  session limit before reporting, but their on-disk edits + new test files
  survive. Recover by: `git diff` the node file, check for the new
  `test_tile_<node>_cutile_runtime.py`, run the GPU tests, finish any incomplete
  work items, then update the logs. Do NOT trust an orchestrator "finished"
  unless its tests actually pass on re-run.

## Current status: Batches A and B COMPLETE & VERIFIED. Batch C NOT STARTED.

| Node          | Status                     | Runtime (GPU) tests |
|---------------|----------------------------|---------------------|
| tile_load     | TESTING-COMPLETE (batch A) | PASS (1 xfail*)     |
| tile_reduce   | TESTING-COMPLETE (batch A) | PASS                |
| tile_iota     | TESTING-COMPLETE (batch A) | PASS                |
| tile_store    | TESTING-COMPLETE (batch B) | PASS (56)           |
| tile_binop    | TESTING-COMPLETE (batch B) | PASS (1 skip)       |
| tile_mask_gen | TESTING-COMPLETE (batch B) | PASS                |
| tile_mma      | **NOT STARTED (batch C)**  | —                   |
| tile_unop     | **NOT STARTED (batch C)**  | —                   |
| tile_ite      | **NOT STARTED (batch C)**  | —                   |

**Last verified full run (2026-06-25, post-cleanup):**
`pytest tests/passes/vectorization/lib_nodes/` = **673 passed, 4 skipped, 1 xfailed**
(187s). This is the green baseline batch C must preserve.

The 1 remaining xfail is `test_strided_1d_stride2` in
`test_tile_load_cutile_runtime.py` — a **load-side** gather `[:]`-subscript
limitation (strided 1-D `B[i]=A[2*i]`), timeboxed and intentionally left as a
strict xfail with a precise reason. NOT a batch-C item unless someone wants to
take it on as a stretch.

## What landed in Batch B (the work just completed)

### tile_store (`nodes/tile_store.py`, +108/−19; recovered from interruption)
- **WI1**: K<ndim scatter shape fix — store-side `__idx{k}` expanded to
  ndim-rank with singleton dims. This un-blocked 3 inherited load xfails.
- **WI2**: WCR / atomic scatter — removed the `NotImplementedError`; maps
  reduction type → `{Sum: ct.atomic_add, Min: ct.atomic_min, Max: ct.atomic_max}`
  (all three confirmed present in cuTile 1.4.0). Masked WCR uses
  `ct.where(mask, upd, 0)` (atomics have no `mask=`). Other reductions raise a
  clear `NotImplementedError`.
- **WI3**: int64 index dtype via the (now shared) `needs_int64` helper.
- **WI4**: aligned path uses `ct.store(order=all_dimensions)` (mirrors load's
  `order=` transpose semantics).
- **WI6**: removed `@pytest.mark.xfail` from the 3 inherited transpose/permute
  load tests (`test_transpose_2d_k1_xfail`, `test_permute_3d_k1_xfail`,
  `test_permute_3d_k2_xfail`) — they now PASS. (Function names still contain
  `_xfail` and docstrings still say "fails at runtime" — **cosmetic stale
  naming**, a cheap cleanup for whoever touches that file next.)
- New `test_tile_store_cutile_runtime.py` (56 tests) — all pass.

### tile_binop (`nodes/tile_binop.py`; all in-node, no shared edits)
- A1 `node.validate()` at expansion start; A2 int-`**` → float-cast-then-back
  (fixes a real cuTile crash "Missing binary arithmetic implementation for pow,
  int"); A3 dtype-aware masked fill (`False` for bool, `0` for numeric — was
  hardcoded `False`); A4 explicit `ct.astype` Symbol-operand promotion; A5
  defensive int-`/` float-cast. Imports `dace_dtype_to_cutile_str`.
- New `test_tile_binop_cutile_runtime.py` (36 GPU) + `test_tile_binop_cutile.py`
  (25 structural). 92 passed, 1 skipped.
- **SKIP** `test_mul_symbol_rhs_free`: cuTile Python backend does not forward
  FREE `dace.symbol`s into `ct.program` kernels (platform limitation; literal
  symbol exprs work fine).

### tile_mask_gen (`nodes/tile_mask_gen.py`; + ONE shared-file edit)
- `_mask_needs_int64(global_ubs)` (symbolic OR >2^31-1 → int64); three guards in
  the expansion (validate each ub expr, trailing-K grid-axis binding check,
  dynamic `ct.arange` dtype); by-design docs for the `<`-only OOB-tail form +
  unused `iter_vars`.
- New `test_tile_mask_gen_cutile_runtime.py` (51 tests). 122 passed.

### Coordinator post-batch-B cleanup (serial refactor, DONE & verified)
- Consolidated the duplicated `_needs_int64` (was a local copy in BOTH
  tile_load.py and tile_store.py) into **`_pure_codegen.needs_int64(desc, coeffs)`**
  + `_INT32_MAX`. Both nodes now import it; the store unit tests were repointed
  to `from dace.libraries.tileops._pure_codegen import needs_int64 as _needs_int64`.
- mask_gen's `_mask_needs_int64` was **left in place** — it has a different
  signature (takes ub-expr strings, not desc+coeffs). Its local `_INT32_MAX` is a
  trivial duplicate; not worth folding.

## Shared-file ownership (to avoid races in batch C)

Shared/hot files edited so far — keep batch-C edits minimal and logged:
- `_pure_codegen.py` — hosts `validate_cutile_expr` + `_CPP_PATTERN`,
  `needs_int64` + `_INT32_MAX`, and the cuTile helpers (`cutile_grid_dim_offset`,
  `cutile_tile_dim_bids`, `gather_lane_offset`, `offset_via_strides`,
  `resolve_gather_deps`, `tile_offset`, `nested_loops`, packed-layout validators).
- `_cutile_dtypes.py` (NEW, batch A) — `_DACE_TO_CUTILE_DTYPE` +
  `dace_dtype_to_cutile_str`. bool→`ct.bool_`, all float/int widths.
- `cutile_target.py` (batch A) — `_needs_gather_scatter` routes dim-mismatch
  (tile dims < map dims) through gather/scatter.
- `test_cutile_expansions.py` — a couple of assertions bumped (int32→int64).

## Batch C — what's known going in (carry-over notes)

- **tile_unop** (`nodes/tile_unop.py`): has the **SAME hardcoded-`False`
  masked-fill bug** that binop's A3 fixed — apply the dtype-aware fill fix here.
  Likely also wants the same `node.validate()` + dtype-promotion treatment.
- **tile_mma** (`nodes/tile_mma.py`): the biggest gap — **NO runtime tests at
  all**, and **uniform-dtype-only** (mixed-precision MMA is the primary
  robustness target, e.g. fp16×fp16→fp32 accumulate). Needs end-to-end GPU
  tests vs NumPy from scratch. Verify the cuTile MMA API surface empirically
  (`import cuda.tile as ct`; check `ct.mma`/`ct.dot` naming, accumulator dtype
  rules, supported operand dtype combos) before planning.
- **tile_ite** (`nodes/tile_ite.py`): select/where node. Check masked-fill /
  dtype-promotion parity with binop; verify `ct.where` dtype coercion on mixed
  branches; cover all-true/all-false/partial predicates + multi-dim K≥2.
- For each: spawn a coding-orchestrator, have it research → return a "PLAN:"
  (+ any "DECISION NEEDED:") before implementing, resolve implementation-scope
  decisions at the coordinator level, relay only genuinely big ones to the human.

## cuTile API facts already verified (don't re-discover)

- `order=` IS a real transpose, applied to ARRAY axes BEFORE the tile space is
  built → `index` AND `shape` must be in permuted order, e.g.
  `ct.load(A,(j,i),shape=(tn,tm),order=(1,0))`. Post-load alt: `ct.transpose`/`ct.permute`.
- `ct.atomic_add` / `ct.atomic_min` / `ct.atomic_max` all exist (tuple indices,
  `check_bounds` default True, **NO `mask=`** → gate with `ct.where`).
- `ct.scatter(mask=)` and `ct.store(order=)` confirmed working on device.
- `ct.bool_` is the bool type name (NOT `ct.bool`).
- `**` on int32 operands crashes → cast to float first.
- Both `ct.store` and `ct.scatter` silently drop OOB writes.
- cuTile Python backend does NOT forward free `dace.symbol`s into kernels.
- Installed cuda-tile version: **1.4.0**.

## Environment / standing constraints (verbatim — must hold)

- Run **all** commands with CWD = `/workspace/wt/improve-libnodes/dace`.
  Test paths are repo-relative: `pytest tests/passes/vectorization/lib_nodes/...`.
- `/venv/main/bin/python -B -m pytest ...` (always `-B`).
- GPU present: NVIDIA RTX 5060 Ti, 16 GB. `import cuda.tile as ct` works.
- **Do NOT `pip install -e`** (shared read-only venv; breaks other agents).
- **Do NOT do state-changing git ops unless explicitly asked.** (The human DID
  ask for the batch-B commit — see PROGRESS.md activity log / git log.)
- `from __future__ import annotations` is FORBIDDEN (breaks DaCe introspection).
- CuTile widths must be powers of 2; shapes are compile-time constants.
- Mark CuTile runtime tests `@pytest.mark.gpu`. Use
  `InstrumentationType.PythonTimer` (not `Timer`). Testing must include full
  end-to-end integration tests, not just structure checks.

## Exact resume steps for the next agent

1. Read this file + `PROGRESS.md`. Confirm the green baseline:
   `/venv/main/bin/python -B -m pytest tests/passes/vectorization/lib_nodes/ -q`
   → expect 673 passed, 4 skipped, 1 xfailed.
2. Get the human's go for batch C (the human paused AT the batch-B→C gate; this
   hand-off + commit IS the pause).
3. On go: spawn 3 coding-orchestrators (tile_mma, tile_unop, tile_ite), brief
   each with the carry-over notes above, collect PLANs, resolve decisions, let
   them implement to passing GPU tests, recover/verify, update logs, then the
   final batch-C go/no-go = effort complete.

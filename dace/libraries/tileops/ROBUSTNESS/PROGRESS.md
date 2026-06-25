# cuTile Expansion Robustness — Master Progress Tracker

> **If you are a new/continuing agent: START HERE.** This directory is the single
> source of truth for the "make cuTile expansions robust" effort. Read this file,
> then `README.md`, then `SHARED_FINDINGS.md` and the relevant `node_*.md`.

## Goal

Make the **cuTile expansions** of every tileops library node robust enough to handle
*arbitrary programs and arbitrary combinations of operations*. For example a load may
combine an indirect gather + strided access + transposed axes; an MMA may combine
mixed dtypes and transposed operands.

- **Priority 1 (hard requirement): correctness**, verified by **GPU runtime tests**
  (`@pytest.mark.gpu`) that run end-to-end (`@dace.program` → canonicalize/vectorize →
  cuTile expansion → compile → run on GPU → compare against NumPy).
- **Priority 2: speed** (only after correctness is proven).

## Environment facts (verified 2026-06-24)

- Run **all** commands with CWD = `/workspace/wt/improve-libnodes/dace`.
  Test paths are repo-relative: `pytest tests/passes/vectorization/lib_nodes/...`.
- Python: `/venv/main/bin/python -B -m pytest ...` (always `-B`).
- GPU present: **NVIDIA RTX 5060 Ti, 16 GB**. `import cuda.tile as ct` works in `/venv/main`.
- Branch: `agent/improve-libnodes`. Do NOT do state-changing git ops unless asked.
- Do NOT `pip install -e` (shared read-only venv; would break other agents).

## How this effort is run

A top-level **coordinator** (Claude) spawns one **coding-orchestrator** subagent per
node, talks to it as if it were the user, relays *big* decisions to the human, and keeps
these notes current. Orchestrators' own AskUserQuestion may not surface to the human —
the coordinator is the bridge. See `README.md` for the protocol.

## The 9 nodes

`tile_load`, `tile_store`, `tile_mma`, `tile_binop`, `tile_unop`, `tile_reduce`,
`tile_ite`, `tile_iota`, `tile_mask_gen`.

## Status table

Status values: NOT STARTED · RESEARCHING · PLANNING · IMPLEMENTING · TESTING · BLOCKED(human) · DONE

| Node          | Status      | Runtime (GPU) tests | Owner agent | Last update |
|---------------|-------------|---------------------|-------------|-------------|
| tile_load     | TESTING-COMPLETE (batch A) | PASS (5 xfail*) | coordinator | 2026-06-24  |
| tile_reduce   | TESTING-COMPLETE (batch A) | PASS         | coordinator | 2026-06-24  |
| tile_iota     | TESTING-COMPLETE (batch A) | PASS         | coordinator | 2026-06-24  |
| tile_store    | TESTING-COMPLETE (batch B) | PASS (56) | coordinator | 2026-06-24  |
| tile_binop    | TESTING-COMPLETE (batch B) | PASS (1 skip) | coordinator | 2026-06-24  |
| tile_mask_gen | TESTING-COMPLETE (batch B) | PASS  | coordinator | 2026-06-24  |
| tile_mma      | NOT STARTED (batch C)  | —        | —           | —           |
| tile_unop     | NOT STARTED (batch C)  | —        | —           | —           |
| tile_ite      | NOT STARTED (batch C)  | —        | —           | —           |

## Decision log (human-in-the-loop)

Record every decision relayed to/from the human here, with date.

- 2026-06-24 — Effort kicked off. Coordinator set up note system, launched background
  surveys (current code state + cuTile API). Strategic question (execution model +
  ordering) relayed to human; awaiting answer.
- 2026-06-24 — **Human decision**: execution model = **small parallel batches**;
  cadence = **go/no-go between batches** (autonomous within a batch to passing GPU
  tests; coordinator summarizes + gets go-ahead before the next batch).
  Batches: **A = {tile_load, tile_reduce, tile_iota}**, B = {tile_store, tile_binop,
  tile_mask_gen}, C = {tile_mma, tile_unop, tile_ite}.
  Mitigations for shared-file conflicts: each orchestrator owns its node file + puts
  NEW GPU runtime tests in a node-specific file (`test_tile_<name>_cutile_runtime.py`);
  edits to hot shared files (cutile_target.py, _pure_codegen.py, _dispatch.py) kept
  minimal and logged in node_*.md so the coordinator can serialize if they collide.

## Activity log (most recent first)

- 2026-06-25 — **Batch B COMPLETE & VERIFIED.** tile_store recovered from session-limit interruption
  (same pattern as batch A): on-disk diff (+108/-19) + new `test_tile_store_cutile_runtime.py` (56
  tests) survived. All 6 WIs landed: WI1 store-side scatter-index expansion, WI2 WCR via
  `ct.atomic_add/min/max` (all confirmed in cuTile 1.4.0), WI3 local `_needs_int64`, WI4 `order=`
  aligned path, WI5 tests, WI6 flipped the 3 inherited load xfails. RESULTS: store runtime = 56
  passed; 3 formerly-xfailed load tests now PASS; `test_strided_1d_stride2` still xfailed (timeboxed,
  load-side). Full `tests/passes/vectorization/lib_nodes/` = **673 passed, 4 skipped, 1 xfailed**.
- 2026-06-25 — **Coordinator post-batch-B cleanup DONE (serial).** Consolidated the duplicated
  `_needs_int64` (tile_load + tile_store) into `_pure_codegen.needs_int64(desc, coeffs)` + `_INT32_MAX`;
  both nodes now import it; store unit tests repointed. (mask_gen's `_mask_needs_int64` kept — different
  signature: takes ub-expr strings.) Re-ran load+store = 155 passed, 1 xfailed. **AWAITING HUMAN
  GO/NO-GO before batch C = {tile_mma, tile_unop, tile_ite}.**

- 2026-06-24 — **All 3 batch-B plans approved; orchestrators implementing.** Coordinator resolved
  all decisions (none needed the human — all implementation-scope):
  - tile_mask_gen: hoist `validate_cutile_expr`+`_CPP_PATTERN` to `_pure_codegen.py` (done); include
    combination tests. (DONE in part — see test_tile_iota_cutile_robustness.py now imports from
    `_pure_codegen`.)
  - tile_binop: negative-int `/`/`%` parity = NON-ISSUE (both backends agree / frontend promotes);
    found+fixing a real `**`-on-int32 cuTile crash; A1–A5 in tile_binop.py only.
  - tile_store: store-side scatter-index expansion (un-xfails 3), WCR via `ct.atomic_add` (+min/max
    if API has them), local `_needs_int64`, `order=` on aligned path, flip inherited xfails.
  - **Coordinator TODO (serial, post-batch-B): consolidate duplicated `_needs_int64` into
    `_pure_codegen.py`** (tile_load + tile_store will each have a local copy).
  - Cross-batch file ownership: only tile_mask_gen edits `_pure_codegen.py` + `tile_iota.py`;
    binop & store stay in their own node files + test files. No `cutile_target.py`/`_dispatch.py` edits.

- 2026-06-24 — **Human GO for batch B.** Launched batch-B orchestrators (background). Agent IDs:
  tile_store = `a35b3145ad5d04d4e`, tile_binop = `ac11ef06a4fce3a18`,
  tile_mask_gen = `a66cadd0ca9d65309`. (Session-scoped; resume via SendMessage within-session.)
  Each briefed to research → return "PLAN:" (+ "DECISION NEEDED:") before implementing.
  Flagged cross-cutting items: tile_store must fix the 5 inherited K<ndim transpose/permute
  scatter xfails + WCR/atomic_add; tile_binop must surface a negative-int `/`/`%` parity decision;
  tile_mask_gen must surface a decision on hoisting `_validate_cutile_expr` to a shared module.

- 2026-06-24 — **Batch A RECOVERED & VERIFIED after session-limit interruption.** The batch-A
  orchestrators were killed by the session limit before reporting, but their on-disk changes
  survived. Coordinator inspected the diffs, fixed stale mid-refactor unit tests (iota dtype map),
  and re-ran everything. RESULTS: full `tests/passes/vectorization/lib_nodes/` =
  **501 passed, 3 skipped, 5 xfailed**; iota unit robustness = **43 passed**. Changes recovered:
  - tile_load: int64 index dtype (`_needs_int64`), transposition via `order=` (settled semantics,
    replaces post-load `ct.permute`), `mask=` on gather paths.
  - tile_reduce: correct identity literals via numpy introspection, `ct.astype` positional args,
    genuine `_mask` consumption.
  - tile_iota: dtype-aware `ct.arange`, C++-expr validation guard.
  - NEW shared module `dace/libraries/tileops/_cutile_dtypes.py` (DaCe→cuTile dtype map).
  - SHARED `cutile_target.py`: `_needs_gather_scatter` now routes dim-mismatch
    (tile dims < map dims, e.g. axis reduction) through gather/scatter — unblocked reduce runtime.
  - 5 xfails (strict) = K<ndim transpose/permute *scatter* shape mismatch → deferred to batch B.
  **AWAITING HUMAN GO/NO-GO before batch B.** See node_tile_{load,reduce,iota}.md for detail.

- 2026-06-24 — **`order=` semantics SETTLED** (after a brief flip-flop with the human +
  reading the full installed docstring). FINAL: `order=` IS a real transpose/permute, but
  it's a "permutation applied to ARRAY axes BEFORE the tile space is constructed" — so
  `index` AND `shape` must be in the permuted order (e.g. `ct.load(A,(j,i),shape=(tn,tm),
  order=(1,0))`). It is NOT a post-load tile permute. The post-load alternative is
  `ct.transpose(x,axis0,axis1)` / `ct.permute(x,axes)`. Both valid; `order=` is TMA-friendly
  but error-prone, post-load is simpler + a shuffle. Notes corrected (SHARED_FINDINGS.md,
  node_tile_load.md, banner atop CUTILE_API_NOTES.md); tile_load orchestrator re-briefed with
  the settled rule + mandatory 2-D-transpose & ≥3-D-permute GPU tests vs NumPy.
- 2026-06-24 — (superseded) initial over-correction had claimed `order=` was memory-layout-only.

- 2026-06-24 — Launched **batch A** orchestrators (background, this session). Agent IDs:
  tile_load = `a2dfb989e6632c83b`, tile_reduce = `ab8f94f7fe4b00650`,
  tile_iota = `a49609269ae2e5862`. (IDs are session-scoped; resume via SendMessage
  within-session only.) Coordinator owns the `node_*.md` logs (orchestrators lack Write).
- 2026-06-24 — Both background surveys completed: `CURRENT_STATE.md` (per-node gaps,
  file:line) and `CUTILE_API_NOTES.md` (cuda-tile 1.4.0 API). Key facts distilled into
  `SHARED_FINDINGS.md`.
- 2026-06-24 — Created ROBUSTNESS/ note system. Surveyed layout: 9 nodes, existing
  tests in `tests/passes/vectorization/lib_nodes/`. GPU + cuda.tile confirmed working.

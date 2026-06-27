# Handoff — Python/cuTile expansions for CopyLibraryNode & MemsetLibraryNode

**Branch:** `copy-nodes-2` (was `agent/copy-nodes`)
**Date:** 2026-06-27
**Scope:** Make the DaCe Python/cuTile code-generation backend able to *consume* the
newly-merged explicit copy/memset library nodes, so that the `InsertExplicitCopies`
pass is safe to run on cuTile (Python-backend) SDFGs.

---

## 1. Problem this solves

The merged `explicit-copy-memset-nodes` feature adds `CopyLibraryNode` /
`MemsetLibraryNode` (`dace/libraries/standard/nodes/`) plus the
`InsertExplicitCopies` pass, which lifts implicit `AccessNode→AccessNode` copy
*edges* into explicit copy library *nodes*. It fires only on `_STANDARD_STORAGES`
(CPU_Heap, GPU_Global, GPU_Shared, Register, …); `CuTile_Tile` is excluded, so the
`ct.load`/`ct.store` tile path is unaffected.

The collision: code generation order is

```
dace/codegen/codegen.py
  :222  sdfg.expand_library_nodes()     # global, unconditional
  :276  target.preprocess(sdfg)         # AFTER expansion
  :284  # SDFG frozen from here
```

So any copy/memset node present at codegen entry is expanded **first**, via its
`implementation` (default `'Auto'`). For the Python backend the pre-existing
expansions were unusable:

- `CopyLibraryNode.Auto` routed multi-element `CPU_Heap↔GPU_Global` to
  `MemcpyCUDA1D/2D/NDStrided` — **C++/CUDA only** (`cudaMemcpyAsync` tasklets);
  `ExpandMappedTasklet` *raises* on the CPU/GPU boundary.
- Result: running `InsertExplicitCopies` on a cuTile SDFG broke codegen (a C++
  memcpy tasklet got emitted as Python).

Because the contract is *expand-then-generate*, the framework-correct fix is a
**new expansion** (not a target `preprocess` hook — that runs too late and the SDFG
is frozen afterward).

## 2. What was implemented

Purely additive, **gated behind `is_python_backend()`** — the default C++/CUDA path
is byte-for-byte unchanged.

| File | Change |
|---|---|
| `dace/libraries/standard/helper.py` | New `is_python_backend(parent_state)` → `parent_state.sdfg.root_sdfg.backend == BackendLanguage.Python` (safe `except (AttributeError, IndexError, RuntimeError) → False`). |
| `dace/libraries/standard/nodes/copy_node.py` | New `ExpandPython` expansion + `'Python'` in `implementations`; `select_copy_implementation` early-returns for the Python backend. |
| `dace/libraries/standard/nodes/memset_node.py` | New `ExpandPython` expansion + `'Python'` in `implementations`; `select_memset_implementation` early-returns for the Python backend. |

**Design — copy:** `ExpandPython` builds a wrapper SDFG (via
`_make_expansion_sdfg(allow_cross_storage=True)`) containing a single bare
`AccessNode→AccessNode` edge with a full-array memlet. The inner descriptors
preserve the outer storages, so the Python backend's existing `copy_memory`
dispatcher lowers the inner edge to:
- `.set()` for host→device (`CPU_Heap→GPU_Global`),
- `.get(out=...)` for device→host (`GPU_Global→CPU_Heap`),
- plain assignment for same-storage (CPU→CPU, GPU→GPU).

Selector routing on the Python backend: single-element same-storage → `'Tasklet'`;
everything else → `'Python'`.

**Design — memset:** `ExpandPython` returns a bare `_out = 0` Tasklet for
single-element **or GPU-resident** arrays (the Python backend emits `arr[...] = 0`
via CuPy/NumPy broadcasting — a host-side `Sequential` map cannot touch device
memory and would fail validation); for **CPU multi-element** it builds a
`Sequential` mapped-tasklet SDFG. Selector: single-element → `'tasklet'`, else
`'Python'`.

Both guardrails from the design review held with no fallback needed:
1. The inlined NestedSDFG's inner AN→AN edge really does dispatch to `.set()/.get()`
   (storages preserved through the wrapper) — proven by GPU cross-storage +
   `InsertExplicitCopies` integration tests.
2. The plain Python target already handles same-storage CPU→CPU and GPU→GPU AN→AN
   copies — confirmed by execution tests.

## 3. Tests

New (executed, including GPU):
- `tests/library/test_copy_memset_python_expansion.py` — 42 CPU tests
  (helper / registration / selector / expansion structure / compile+run vs NumPy:
  1D/2D/3D, slices, single-element, symbolic sizes, float32/int32).
- `tests/library/copy_memset_python_gpu_test.py` — 24 `@pytest.mark.gpu` tests
  (cross-storage H2D/D2H/roundtrip, same-storage D2D, GPU memset, `InsertExplicitCopies`
  Python-backend integration, edge cases — odd shapes, 3D, single-element, int64).

Results (CWD = `/workspace/wt/copy-nodes/dace`, `/venv/main/bin/python -B -m pytest`):

| Suite | Result |
|---|---|
| New copy/memset Python+GPU tests | 66 passed |
| `tests/library/copy_node_test.py` + `memset_node_test.py` + `passes/insert_explicit_copies_test.py` | 144 passed (C++/CUDA regression — unchanged) |
| `tests/codegen/python_backend/` | 681 passed, 1 skipped, 9 xfailed |

**Zero regressions** attributable to this change.

## 4. Pre-existing issue (NOT introduced here — do not chase as part of this work)

`tests/canonicalize/` has **31 failures + a teardown segfault** that predate this
branch. Proven independent: they run on the **default C++ backend** (so
`is_python_backend()` returns False and none of the new code runs), and a
representative failure reproduces on the untouched baseline tree
(`/workspace/dace`, `vectorized-tile-ir`) as
`TypeError: Passing an array to a scalar (type int) in argument "act"` at
`dace/data/ctypes_interop.py:76`. The same interop fault makes the broad
`tests/passes/vectorization/` sweep SIGABRT during failure teardown.
→ Separate bug in core `ctypes_interop`, orthogonal to copy/memset work.

## 5. Open items / possible follow-ups

1. **GPU memset performance:** the bare-Tasklet path emits `arr[...] = 0`
   (correct, CuPy broadcast) rather than `cudaMemsetAsync`. Fine for cuTile SDFGs
   where memset is rare; could be specialized later.
2. **0-D NestedSDFG for single-element cross-storage copies:** `ExpandPython` makes
   0-D inner arrays via `collapse_shape_and_strides`; works today via `.set()/.get()`
   but worth watching if shapes get exotic.
3. **Pre-existing `ExpandTasklet` naming bug** (`memset_node.py`, the `'tasklet'`
   expansion): a misnamed `inp` variable — present before this change, left
   out of scope. Trivial to fix if touched.
4. **Adopting `InsertExplicitCopies` in `VectorizeCuTile`:** not done. The backend
   now *tolerates* copy nodes; actively inserting them in the pipeline would
   round-trip the host↔device copies the backend already handles natively, so it
   only pays off if a later pass needs to reason over explicit copy nodes. Decide
   per need.

## 6. Where things live

- Code: the 3 files in §2.
- Tests: the 2 files in §3.
- Full session report (TileIR repo, not this repo):
  `TileIR/ai_session_reports/2026-06-27/python-backend-copy-memset-expansions.md`.
- Background: see `dace/codegen/py/cutile_target.py` `copy_memory()` /
  `_emit_cross_storage_copy()` for the `.set()/.get()` emission this expansion
  lowers to; `dace/transformation/passes/vectorization/vectorize_cutile.py` for the
  cuTile pipeline that sets `sdfg.backend = Python` before codegen.
</content>
</invoke>

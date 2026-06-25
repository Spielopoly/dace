# node: tile_load — running log

**Status:** TESTING-COMPLETE (batch A, all GPU runtime tests pass — awaiting human go/no-go)
**File:** `dace/dace/libraries/tileops/nodes/tile_load.py` (683 lines)
**New runtime tests:** `tests/passes/vectorization/lib_nodes/test_tile_load_cutile_runtime.py`

## Current state & gaps (from CURRENT_STATE.md)
Most complete node. Gaps:
- packed-layout-only (`NotImplementedError` for non-packed).
- `prod` reduction has no load-time pad identity (boundary fill).
- masked-gather relies on unconfirmed `ct.gather(mask=)` API.
- index tiles hardwired int32.

## Edge-case matrix to cover (robustness target)
- contiguous / strided / transposed-permuted axes — TWO valid mechanisms: (1) `order=` on the
  load (permutes array axes BEFORE tile space → emit index/shape in the permuted order too), or
  (2) post-load `ct.transpose`/`ct.permute`. Test both + combinations; assert vs NumPy.
- indirect gather (tuple index tiles), incl. gather + strided + transposed together.
- non-divisible sizes → boundary/remainder via `padding_mode` (not synthesized masks); verify OOB lanes.
- all-true / all-false / partial masks; padding_value correctness.
- multi-dim K≥2; symbolic sizes (`dace.symbol`); mixed dtypes incl. non-int32 index tiles.

## Changes made (recovered from on-disk diff 2026-06-24)
- **int64 index dtype**: new `_needs_int64(src_desc, coeffs)` — index tiles (`ct.arange`,
  gather index entries) emit `ct.int64` when a source dim × stride coeff could exceed the
  int32 range, or the size is symbolic (conservative). Fixes the "index tiles hardwired int32" gap.
- **Transposition via `order=`** (settled semantics): the aligned-contiguous path now builds
  `index`/`shape` in the permuted `all_dimensions` order and passes `order={all_dimensions}`
  to `ct.load`, instead of the old post-load `ct.permute`. Mask gating on this path via `ct.where`.
- **gather `mask=`**: gather/general paths now pass `mask=_mask` to `ct.gather` directly.
- `src_begins` dict refactor centralizes the per-source-dim "memlet begin" index used for
  unused/replicated source dims.

## Tests (commands + results)
- New file `tests/passes/vectorization/lib_nodes/test_tile_load_cutile_runtime.py` (58 KB).
- `pytest tests/passes/vectorization/lib_nodes/test_tile_load_cutile_runtime.py` → **PASS**
  (part of the 164-passed batch-A run; 5 xfail total across batch A).

## Decisions needed / open issues
- **xfail (strict, documented)**: K<ndim transpose/permute fails at runtime with a
  *scatter shape mismatch* (tile shape `(1,W)` vs index shape `(W,)`). These are
  STORE-side (scatter) limitations — `test_transpose_2d_k1_xfail`,
  `test_permute_3d_k1_xfail`, `test_permute_3d_k2_xfail`. Defer to **batch B (tile_store)**;
  load-side transposition itself works (K==ndim cases pass).

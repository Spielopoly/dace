# Shared findings — cross-node knowledge index

This file collects knowledge that applies across nodes. Detailed dumps live in the
two companion files; summarize key takeaways here as they solidify.

- **Current code state of each node + gaps:** see `CURRENT_STATE.md`.
- **cuTile-Python API facts (load/store/mma/reduce/...):** see `CUTILE_API_NOTES.md`.

## cuTile API quick-reference (installed: cuda-tile 1.4.0) — full detail in CUTILE_API_NOTES.md

- **Hard constraints**: all tile dims are compile-time **power-of-2** constants; tiles are
  **immutable**; `ct.arange` is the 1-D iota (dtype keyword-only); positive loop steps only.
- **Does NOT exist**: `ct.iota`, `ct.select`, `ct.dot`. Use `ct.arange`, `ct.where`, 1-D `ct.matmul`.
- **load**: set `padding_mode` (ZERO/NAN/±INF/…) for boundary/remainder fills instead of
  synthesizing mask tiles; default `UNDETERMINED` is *garbage* on OOB lanes — always set it
  when a tile can straddle a boundary. **`order=` DOES express transpose/permute** but as a
  "permutation applied to ARRAY axes BEFORE the tile space is constructed" — so `index` AND
  `shape` must be given in the permuted order too (transposed load =
  `ct.load(A,(j,i),shape=(tn,tm),order=(1,0))`). Alternative = post-load tile-space permute
  `ct.transpose`/`ct.permute` (also `Tile.transpose`/`.permute`). `order=` reads transposed
  directly (TMA-friendly) but is error-prone; verify on GPU vs NumPy. See CUTILE_API_NOTES banner.
- **store**: partial OOB stores are **silently dropped** (that's the write-remainder story);
  0-d/scalar tiles store by broadcast; `order=` here is the same array-axis permutation as load (see load/banner).
- **gather/scatter**: tuple-of-broadcastable index tiles + `mask`, `padding_value`, `check_bounds=True`.
  **Negative indices = OOB (NOT Python wraparound).** Accumulating scatter → use **`atomic_add`** (bulk, race-free), not `scatter`.
  `load_advanced_indexing` = sparse-rows × dense-`Slice` window (cheaper structured gather).
- **reduce**: built-ins `sum/prod/max/min` take an **axis tuple + keepdims** (argmax/argmin: int axis only);
  `ct.reduce`/`ct.scan` accept tuple payloads + custom `func`/`identity` for arbitrary WCR;
  `rounding_mode`/`flush_to_zero` for numeric parity.
- **mma**: computes `(x@y)+acc`; **output dtype = acc dtype**; inputs are **NOT promoted** (unlike `matmul`).
  Pick acc dtype from the input→acc table (i8⇒i32, bf16⇒f32, f16⇒f16/f32). No transpose flag —
  realize transposed operands either via `order=` on the source load (array-axis permute — mind index/shape) or via post-load `ct.transpose`/`ct.permute`. `mma_scaled` exists (Blackwell microscaling).
- **misc**: `ct.where` is the masked-select primitive; `astype` (value cast) vs `bitcast` (raw);
  `reshape(-1)` flattens; `cat` needs equal-shape pairs.

## Headline per-node gaps (2026-06-24 survey — detail + file:line in CURRENT_STATE.md)

- **tile_load** — most complete; packed-layout-only (`NotImplementedError`), `prod` has no load-time pad identity, masked-gather uses unconfirmed `ct.gather(mask=)`, index tiles hardwired int32.
- **tile_store** — WCR/atomic-scatter raises `NotImplementedError` (tile_store.py:152) → all accumulating/collapse stores blocked; non-full-tile structured stores refused; aligned `ct.store` can write OOB on non-divisible sizes.
- **tile_mma** — uniform-dtype only (no mixed precision), no transpose flags, exactly 2D/3-tuple; **NO runtime AND no pure test**.
- **tile_binop** — Python `/` `%` differ from CPP ref for negative ints (TODO 286/288, correctness hazard); masked inactive lanes hardcoded `False`/0 (not op-aware); no `ct.astype` promotion.
- **tile_unop** — same masked-`False`/no-promotion issues; weakest tested (only `neg` runs e2e; 9 ops untested).
- **tile_reduce** — dead docstring promising a `ct.where` fallback never implemented; raw `axis` passthrough; full-reduction→scalar seam unverified on device (runtime deferred — was blocked on a `MarkTileDims` bug).
- **tile_ite** — `_CT_HAS_WHERE` probe computed but unused (dead); uniform-dtype only; no explicit bool cast on non-bool condition.
- **tile_iota / tile_mask_gen** — both emit `expr`/`global_ubs` strings verbatim as Python (CPP-flavored input = silent crash trap); `ct.arange` hardwired int32; mask_gen only supports `<`-OOB form, assumes K tiled dims are trailing grid axes.

## Batch plan (human-approved 2026-06-24)

- **Batch A** (in flight / next): tile_load, tile_reduce, tile_iota
- **Batch B**: tile_store, tile_binop, tile_mask_gen
- **Batch C**: tile_mma, tile_unop, tile_ite
- Go/no-go between batches.

## Cross-node conventions

- cuTile widths must be powers of 2; tile shapes are compile-time constants.
- cuTile expansion is AccessNode-centric (loads/stores at `StorageType.CuTile_Tile`
  AccessNodes; tiles are immutable; NestedSDFGs → module-level functions returning tiles).
- `CuTileSetImplementations` stamps `target_isa="CUTILE"` / `implementation="cutile"`.
- New GPU runtime tests go in node-specific files: `tests/passes/vectorization/lib_nodes/test_tile_<name>_cutile_runtime.py`, marked `@pytest.mark.gpu`.

## Cross-node gotchas / shared files that cause conflicts

Files multiple nodes touch (coordinate edits to avoid clobbering):
- `dace/dace/libraries/tileops/_pure_codegen.py`
- `dace/dace/libraries/tileops/_isa_codegen.py`
- `dace/dace/libraries/tileops/_dispatch.py`
- `dace/dace/codegen/py/cutile_target.py`
- shared test files under `tests/passes/vectorization/lib_nodes/`
- (more to come)

# cuTile-Python API Notes (for robust `ct` codegen)

Authoritative reference for the DaCe code generator that emits `cuda.tile`
(aliased `ct`) Python kernels. Facts below are cross-checked against the
**installed** package and the official docs.

> ## ⚠️ `order=` semantics (2026-06-24, settled w/ human + full docstring) — READ CAREFULLY
> The load/store `order=` param **IS a real transpose/permute mechanism** — but it is a
> **"Permutation applied to ARRAY axes BEFORE the logical tile space is constructed"**, NOT a
> permute of an already-materialized tile. Values: `'C'`=identity `(0,1,2,…)`,
> `'F'`=reversed `(…,2,1,0)`, or an explicit axis tuple.
> Because the permutation happens before tile-space construction, **`index` AND `shape` must
> also be expressed in the permuted order.** Concretely:
> - default `t = ct.load(A,(i,j),(tm,tn))`  ⇒  `t[x,y] = A[i*tm+x, j*tn+y]`
> - transposed `ct.load(A,(j,i),shape=(tn,tm),order=(1,0))`  ⇒  `t[y,x] = A[i*tm+x, j*tn+y]`
>   (index `(j,i)` and shape `(tn,tm)` are swapped too — that's the "great care" part).
>
> **Two valid mechanisms for a transposed/permuted load:**
> 1. `order=` on the load — array-axis permute before tile space; reads transposed directly
>    (TMA-friendly, no register shuffle) but you must keep index/shape/order consistent.
> 2. Post-load tile-space permute — `ct.transpose(x, axis0, axis1)` (>2-D requires explicit
>    axes) or `ct.permute(x, axes)`; also `Tile.transpose`/`Tile.permute`. Simpler, adds a shuffle.
>
> Prefer (1) for perf when codegen can emit permuted index/shape; (2) is a fine fallback.
> ALWAYS verify the chosen mechanism on GPU vs NumPy. (Inline text below predates this note.)

- **Installed version:** `cuda-tile` **1.4.0** ("CUDA Tile Compiler")
  - Module file: `/venv/main/lib/python3.13/site-packages/cuda/tile/__init__.py`
  - Homepage / repo: <https://github.com/nvidia/cutile-python>
- **Docs root:** <https://docs.nvidia.com/cuda/cutile-python/>
  - Operations: <https://docs.nvidia.com/cuda/cutile-python/operations.html>
  - Data model: <https://docs.nvidia.com/cuda/cutile-python/data.html>
  - Execution model: <https://docs.nvidia.com/cuda/cutile-python/execution.html>
  - Performance tuning: <https://docs.nvidia.com/cuda/cutile-python/performance.html>
  - Memory model: <https://docs.nvidia.com/cuda/cutile-python/memory_model.html>

> **Method note.** The public docs HTML is JS-rendered and exposes only one-line
> summaries per op; the *full* signatures, parameter docs, and code examples
> below were extracted from the **installed 1.4.0 docstrings** (the source of
> truth for what actually compiles here). Where the docs and the install agree
> they are noted; mismatches are flagged under "Doc vs install".

---

## 0. Global model facts (constraints that gate everything)

From the data + execution models:

- **Tiles are immutable.** Every op returns a *new* tile; there is no in-place
  mutation. (Codegen already treats tiles as SSA values — keep it that way.)
- **Every tile dimension must be a compile-time constant that is a power of 2.**
  This is the hard `cuda.tile` runtime constraint. Shapes flow as
  `tuple[const int, ...]`. `ct.Constant[int]` is how a kernel parameter is
  marked compile-time-constant (constant embedding generates a distinct
  compiled kernel per unique value).
- **Scalars are 0-d tiles** with shape `()`. Numeric literals become *loosely
  typed* constant scalars.
- **Arrays vs tiles:** a *global array* (the kernel's NumPy/CuPy/Torch argument)
  lives in memory with shape+strides; tiles live only inside kernel code.
  Arrays support essentially only load/store/gather/scatter/atomic. Multiple
  array arguments **must not alias**.
- **Tile space:** `load`/`store` partition an array of shape `(M, N)` with tile
  shape `(tm, tn)` into a grid of `(cdiv(M,tm), cdiv(N,tn))` tiles; the *index*
  passed to load/store is a **tile-space index**, not an element index.
- **Parallelism:** grid is 1-3D; `ct.launch(stream, grid, kernel, args)`.
  Inside a kernel `ct.bid(axis)` / `ct.num_blocks(axis)` give block coords
  (axis in {0,1,2}). `ct.num_tiles(array, axis, shape=..., order=...)` gives the
  tile-space extent (uses `cdiv`).
- **Control-flow Python subset:** `if/for/while` allowed but **range step must
  be strictly positive** — no negative-step ranges. (Matches DaCe's
  "Maps must have positive steps" rule — good.)
- **Broadcasting = NumPy rules:** align trailing dims; dims compatible if equal
  or one is 1; left-pad missing dims with 1.

---

## 1. Load / Store (the highest-value section for robustness)

URL: operations.html#load-store

### `ct.load`
```
ct.load(array, /, index, shape, *, order='C',
        padding_mode=PaddingMode.UNDETERMINED,
        latency=None, allow_tma=None,
        memory_order=MemoryOrder.WEAK, memory_scope=MemoryScope.NONE) -> Tile
```
- `array`: the global array to load from.
- `index` (`tuple[int,...]`): **tile-space** index (runtime ints OK). Element
  offset is `index[d] * shape[d]`.
- `shape` (`tuple[const int,...]`): tile shape — **compile-time, power-of-2**.
- **`order`** ('C' | 'F' | `tuple[const int,...]`): permutation of array axes
  *before* the tile space is built — i.e. **transposed / strided loads for
  free**. `'C'` = `(0,1,2,...)` identity; `'F'` = reversed. A tuple lets you
  permute arbitrary axes (e.g. `(0,2,1)` transposes the last two of a 3-D
  array). This is the load-time transpose primitive.
- **`padding_mode`** (`PaddingMode`): fills out-of-bounds elements when a tile
  **partially** extends past the array boundary. **This is the boundary/remainder
  story** — no manual mask tile needed for OOB loads.
  - Values: `UNDETERMINED` (default — garbage; only safe if you *know* you're
    in-bounds), `ZERO`, `NEG_ZERO`, `NAN`, `POS_INF`, `NEG_INF`.
  - If a tile lies *entirely* outside the array → **undefined behavior** (must
    avoid; the grid must not launch fully-OOB tiles).
- `latency` (const int 1..10): DRAM-traffic hint; higher ⇒ deeper prefetch.
  Optional; compiler infers if `None`.
- `allow_tma` (const bool): `False` disables Tensor-Memory-Accelerator path.
  Default allows TMA.
- `memory_order` / `memory_scope`: for ordered loads (`WEAK` default;
  `RELAXED`/`ACQUIRE` valid for loads). Scope only matters when order ≠ WEAK.

Example (boundary fill + transpose):
```python
ct.load(x, (2,), shape=4, padding_mode=ct.PaddingMode.ZERO)  # [8, 9, 0, 0]
ct.load(x, (j, i), shape=(tn, tm), order=(1, 0))             # transposed load
```

### `ct.store`
```
ct.store(array, /, index, tile, *, order='C', latency=None, allow_tma=None,
         memory_order=MemoryOrder.WEAK, memory_scope=MemoryScope.NONE) -> None
```
- `shape` is **inferred from the tile**. Same `index` (tile-space) and `order`
  semantics as `load`.
- **Partial out-of-bounds stores are silently ignored** (the boundary/remainder
  story for stores — again no manual masking required). Fully-OOB tile → UB.
- A **0-d tile / scalar stores by broadcast** to the array rank (handy for
  filling). Otherwise tile rank must equal array rank.
- `memory_order` for stores: `WEAK`/`RELAXED`/`RELEASE`.

### `ct.load_advanced_indexing` / `ct.store_advanced_indexing`
```
ct.load_advanced_indexing(array, indices, /, *, padding_mode=UNDETERMINED,
                          latency=None, allow_tma=None) -> Tile
ct.store_advanced_indexing(array, indices, tile, /, *, latency=None, allow_tma=None) -> None
```
- **Non-contiguous *row/slice* gather** (cheaper & more structured than full
  element gather). `indices` is a tuple of length `array.ndim` where **exactly
  one** entry is a 1-D integer `Tile` (the *sparse* dim, element-space indices)
  and **every other** entry is a `ct.Slice(start, length)` (`start` runtime,
  `length` compile-time power-of-2) describing a contiguous dense range.
- Result shape = the per-dim lengths (sparse-dim length = index-tile length;
  dense dims = their `Slice.length`).
- `padding_mode` fills OOB on *both* sparse and dense dims.
- Use this for "load these N rows, each a contiguous column window" patterns.
```python
row_indices = ct.arange(4, dtype=ct.int32)
tile = ct.load_advanced_indexing(x, (row_indices, ct.Slice(col_start, 4)),
                                 padding_mode=ct.PaddingMode.ZERO)
```

### `ct.gather` / `ct.scatter` (fully general element gather/scatter)
```
ct.gather(array, indices, /, *, mask=None, padding_value=0, check_bounds=True, latency=None) -> Tile
ct.scatter(array, indices, value, /, *, mask=None, check_bounds=True, latency=None) -> None
```
- `indices`: tuple of length `array.ndim`; each entry an integer tile/scalar,
  all **broadcastable to a common shape** — and the **result/store shape is that
  broadcast shape**. (For a 1-D array you may pass a bare tile, equivalent to a
  1-tuple.) This is the multi-dim cartesian-style indexing engine.
- **`mask`** (bool tile/scalar, broadcastable to the common shape): where False,
  gather returns `padding_value` / scatter does nothing.
- **`check_bounds=True`** (default): OOB indices → `padding_value` on gather /
  no-op on scatter. **Effective mask = custom mask AND in-bounds.** Set
  `check_bounds=False` only when indices are provably in-range (OOB ⇒ UB).
- **Negative indices are treated as OOB** (NOT Python-style negative indexing) —
  important: do not rely on `-1` wraparound.
- `padding_value` (gather only): scalar or broadcastable tile, default 0.

### `ct.bid` / `ct.num_blocks` / `ct.num_tiles`
- `ct.bid(axis)`, `ct.num_blocks(axis)` → `int32`, axis ∈ {0,1,2}.
- `ct.num_tiles(array, axis, *, shape, order='C')` → `int32` tile-space extent
  (uses `cdiv`); honors the same `order` permutation as `load`.

**Robustness takeaways (load/store):**
1. Prefer `padding_mode` on `load` + relying on store's silent partial-OOB drop
   for **remainder/non-divisible tiles** instead of synthesizing mask tiles.
2. Use `order=` to express transpose/permutation at load/store time rather than
   materializing `ct.transpose`/`ct.permute` on the tile.
3. For gather/scatter keep `check_bounds=True` and use `mask`/`padding_value`
   instead of clamping indices; never emit negative indices expecting wraparound.
4. Use `ct.load_advanced_indexing` for "sparse rows × dense window" access.

---

## 2. Tile factory / creation

URL: operations.html (Factory)

| fn | signature | notes |
|----|-----------|-------|
| `arange` | `ct.arange(size, /, *, dtype) -> Tile` | values `0..size-1`; `size` const, `dtype` **required keyword**. This is iota. 1-D only. |
| `full` | `ct.full(shape, fill_value, dtype) -> Tile` | shape const power-of-2 tuple. |
| `zeros` | `ct.zeros(shape, dtype) -> Tile` | |
| `ones` | `ct.ones(shape, dtype) -> Tile` | |
| `astile` | `ct.astile(value, dtype) -> Tile` | from nested tuple of scalars; each tuple length must be power-of-2, sibling tuples uniform length. Good for small constant tiles. |

Robustness: there is no N-D iota; build multi-dim index tiles by `arange` +
`expand_dims`/`broadcast_to`, or via `ct.bid` arithmetic.

---

## 3. Shape / dtype manipulation

URL: operations.html (Shape & dtype)

| fn | signature | notes |
|----|-----------|-------|
| `broadcast_to` | `ct.broadcast_to(x, shape) -> Tile` | NumPy broadcasting; target dims power-of-2. |
| `expand_dims` | `ct.expand_dims(x, axis) -> Tile` | insert size-1 axis; also `x[:, None]`. |
| `reshape` | `ct.reshape(x, shape) -> Tile` | **one dim may be `-1`** (inferred). element count preserved. |
| `transpose` | `ct.transpose(x, axis0=None, axis1=None) -> Tile` | 2-D: swaps the two axes if unspecified; **>2-D: axis0/axis1 required**. |
| `permute` | `ct.permute(x, axes) -> Tile` | full axis permutation, `axes` const tuple. |
| `cat` | `ct.cat(tiles, axis) -> Tile` | **pair only**; due to power-of-2 rule **both inputs must have identical shape** (result doubles that axis). |
| `extract` | `ct.extract(x, index, shape) -> Tile` | sub-tile extraction: partitions `x` into a grid of `shape`-sized subtiles; `index` is a **grid index**, not element index (like `load` but on a tile). |
| `astype` | `ct.astype(x, dtype) -> Tile` | value-preserving cast (e.g. int→float). |
| `bitcast` | `ct.bitcast(x, dtype) -> Tile` | raw reinterpret, same bitwidth. |
| `pack_to_bytes` | `ct.pack_to_bytes(x) -> Tile` | flatten → 1-D uint8; total bits must be %8==0. |
| `unpack_from_bytes` | `ct.unpack_from_bytes(x, dtype) -> Tile` | inverse of pack. |

Robustness: `cat`'s equal-shape restriction is a trap — never emit `cat` of
unequal tiles. `reshape(-1)` is the safe way to flatten.

---

## 4. Reductions & scans

URL: operations.html (Reduction, Scan)

### Built-in reductions
```
ct.sum (x, /, axis=None, *, keepdims=False, rounding_mode=None, flush_to_zero=False) -> Tile
ct.prod(x, /, axis=None, *, keepdims=False, rounding_mode=None, flush_to_zero=False) -> Tile
ct.max (x, /, axis=None, *, keepdims=False, flush_to_zero=False) -> Tile
ct.min (x, /, axis=None, *, keepdims=False, flush_to_zero=False) -> Tile
ct.argmax(x, /, axis=None, *, keepdims=False) -> Tile
ct.argmin(x, /, axis=None, *, keepdims=False) -> Tile
```
- **`axis`**: `None` (reduce all → scalar), a const int, **or a tuple of const
  ints** (e.g. `ct.sum(t, (1,2))`) — *except* `argmax`/`argmin` which do **NOT
  accept an axis tuple** (single int or None only).
- **`keepdims`**: keep reduced axis as size-1 (needed to broadcast the result
  back, e.g. softmax denominator).
- `rounding_mode` (`RoundingMode`, float-only, default RN) on sum/prod;
  `flush_to_zero` flushes subnormals — exploit for numeric-parity control.

### Custom reduction / scan
```
ct.reduce(x, /, axis, func, identity, *, keepdims=False)
ct.scan  (x, /, axis, func, identity, *, reverse=False)
```
- `func`: combiner of two 0-d tiles (e.g. `operator.add`, `lambda a,b: a+b`).
  **`x` may be a tuple of N tiles** (multi-payload reduction, e.g. argmax via
  (value,index) pairs); then `func` takes 2N tiles and returns N-tuple, and
  `identity` is an N-tuple of const scalars.
- `identity`: const scalar identity element of `func` (required).
- Use `ct.reduce` for ops not covered by the built-ins (e.g. custom WCR).

### Cumulative scans
```
ct.cumsum (x, /, axis=0, *, reverse=False, rounding_mode=None, flush_to_zero=False) -> Tile
ct.cumprod(x, /, axis=0, *, reverse=False, rounding_mode=None, flush_to_zero=False) -> Tile
```
- Inclusive prefix; `reverse=True` scans from the high end.

Robustness: prefer built-in `ct.sum/max/min/prod` (axis tuple + keepdims) over
hand-rolled `ct.reduce` where the op matches; use `keepdims=True` then
`broadcast_to` for reduce-then-broadcast patterns (softmax, normalization).

---

## 5. MatMul / MMA

URL: operations.html (Matmul)

### `ct.matmul`
```
ct.matmul(x, y, /) -> Tile      # also the `@` operator
```
- LHS/RHS **1-D, 2-D, or 3-D**. 1-D×1-D = dot product (scalar). 3-D = batched;
  **batch dims broadcast** (e.g. (2,2,4)@(4,2)).
- **Supported input dtypes:** `f16, bf16, f32, f64, tf32, f8e4m3fn, f8e5m2, i8, u8`.
- If x,y dtypes differ they are **promoted to a common dtype**; result dtype =
  promoted input dtype.

### `ct.mma` (multiply-accumulate — preferred for K-loop accumulation)
```
ct.mma(x, y, /, acc, *, use_fast_acc=False) -> Tile
```
- Computes `(x @ y) + acc` in one op; **result dtype = dtype of `acc`** (this is
  how you keep a higher-precision accumulator).
- x,y **2-D or 3-D**; batch dims broadcast (same as matmul).
- **dtypes differ ⇒ NOT promoted** (unlike `matmul`!). Caller must match dtypes.
- **Input → Acc/Output dtype table:**
  | Input | Acc/Output |
  |-------|-----------|
  | f16 | f16 or f32 |
  | bf16 | f32 |
  | f32 | f32 |
  | f64 | f64 |
  | tf32 | f32 |
  | f8e4m3fn | f16 or f32 |
  | f8e5m2 | f16 or f32 |
  | i8 / u8 | i32 |
- `use_fast_acc`: fp8-only, Hopper-only (silently ignored elsewhere); trades acc
  precision for throughput.
- **Transposed operands:** there is no `transpose` flag on mma — transpose the
  operand tile first (`.transpose()` / `ct.transpose`) or, better, **load it
  transposed via `order=`**. (See the `mma_scaled` example: `ct.load(...).transpose()`.)

### `ct.mma_scaled` (block-scaled / microscaling, Blackwell sm_100+)
```
ct.mma_scaled(x, x_scale, y, y_scale, /, acc) -> Tile
```
- `result = sum_k x[i,k]*x_scale[i,k//B] * y[k,j]*y_scale[k//B,j] + acc`.
- Scale block size `B = K // K_s`. Allowed (input/scale/B) combos:
  | Input x/y | Scale | Acc/Out | B |
  |-----------|-------|---------|---|
  | f8e4m3fn, f8e5m2 | f8e8m0fnu | f32 | 32 |
  | f4e2m1fn | f8e8m0fnu | f32 | 16, 32 |
  | f4e2m1fn | f8e4m3fn | f32 | 16 |
- Blackwell-only; needs specific tmem scale-factor swizzle layout (see docstring
  example). Likely out of scope for current DaCe expansions; note for future.

Robustness: emit `ct.mma(x, y, acc)` with an `acc` tile of the **correct
output dtype from the table** (e.g. i8 inputs ⇒ i32 acc; bf16 ⇒ f32 acc) and
never assume promotion. Realize transposes via `order=` on the source `ct.load`.

---

## 6. Selection / control flow

URL: operations.html (Selection)

### `ct.where`
```
ct.where(cond, x, y, /) -> Tile
```
- `cond` bool tile of shape `S`; `x`,`y` tile of shape `S`, **same dtype `T`**.
  Returns elementwise select. This is the masked-select / branchless-`if`
  primitive (use for `TileITE`, masked stores via store of a `where` result,
  etc.).
- No `ct.select` exists (it is `where`). cuTile *does* support real Python `if`
  on scalar conditions in tile code, but `where` is the data-parallel form.

---

## 7. Elementwise math / comparison / bitwise (operator-backed)

URL: operations.html (Math / Comparison / Bitwise)

All accept tiles or scalars and broadcast; most have an operator form.

- **Arith:** `add(+) sub(-) mul(*) truediv(/) floordiv(//) mod(%) pow(**)
  cdiv negative(-x) abs`. `add/sub/mul/truediv` take **`rounding_mode` +
  `flush_to_zero`**; `floordiv/mod/pow/cdiv` do not.
- **min/max elementwise:** `minimum(x,y)`, `maximum(x,y)` (+`flush_to_zero`).
- **Transcendental:** `exp exp2 log log2 sqrt rsqrt sin cos tan sinh cosh tanh
  floor ceil isnan`. Several take `rounding_mode`/`flush_to_zero`
  (`exp`, `sqrt`, `rsqrt`, ...). `atan2(x,y)` binary.
- **Comparison** (return bool tile): `greater(>) greater_equal(>=) less(<)
  less_equal(<=) equal(==) not_equal(!=)`.
- **Bitwise:** `bitwise_and/or/xor/not`, `bitwise_lshift/rshift`.

Robustness: dtype follows the promotion rules below; prefer operator forms for
readability, but the named fns give access to `rounding_mode`/`flush_to_zero`
for numeric parity with a CPU reference.

---

## 8. Atomics

URL: operations.html (Atomic)

```
ct.atomic_add(array, indices, update, /, *, check_bounds=True,
              memory_order=ACQ_REL, memory_scope=DEVICE) -> Tile
ct.atomic_cas(array, indices, expected, desired, /, *, check_bounds=True,
              memory_order=ACQ_REL, memory_scope=DEVICE) -> Tile
# also: atomic_and/or/xor, atomic_max/min, atomic_xchg
```
- **Bulk** atomics over `indices` (same indices convention as gather/scatter:
  tuple of broadcastable integer tiles, `check_bounds` honored, negative = OOB).
  Each element is atomic; the batch as a whole is **not** atomic and write order
  is unspecified. Returns the **pre-update** values.
- This is the right lowering for **WCR / scatter-with-accumulate** (e.g.
  `out[idx] += v`) where multiple lanes may target the same address — far safer
  than `scatter` (which would race). Strong robustness lever.

---

## 9. Data types & promotion

URL: data.html

- **dtypes:** `bool_` (8-bit); `uint8/16/32/64`; `int8/16/32/64`;
  `float16/32/64`; `bfloat16`; `tfloat32`; `float8_e4m3fn`, `float8_e5m2`,
  `float8_e8m0fnu`; `float4_e2m1fn`. Each has `.bitwidth` and `.name`.
- **Promotion (binary ops):** category order **bool < integral < float**;
  higher category wins. Same category w/ one constant ⇒ non-constant dtype.
  Same category both non-constant ⇒ a fixed promotion table — and **some combos
  are errors** (e.g. `uint64 + int8` → error). `int32 + float32 → float32`.
- **Loosely-typed constants** (`5`, `3.14`) stay loose until combined with a
  typed tile. Use `ct.int16(5)` etc. for a strictly-typed constant.

Robustness: do not rely on mixed signed/unsigned 64-bit promotion; insert
explicit `ct.astype` where the CPU reference's dtype would differ. `astype`
changes value-preserving; `bitcast` is raw and requires equal bitwidth.

---

## 10. Enums (exact members as installed)

- `PaddingMode`: `UNDETERMINED, ZERO, NEG_ZERO, NAN, POS_INF, NEG_INF`
- `RoundingMode`: `RN` (nearest-even, default), `RZ` (toward zero), `RM`
  (toward -inf), `RP` (toward +inf), `FULL`, `APPROX`, `RZI`.
- `MemoryOrder`: `WEAK` (default), `RELAXED`, `ACQUIRE`, `RELEASE`, `ACQ_REL`.
- `MemoryScope`: `NONE`, `BLOCK`, `DEVICE`, `SYS`.

---

## 11. Metaprogramming / debug / launch (codegen plumbing)

- `ct.static_assert(condition, message=None)`, `ct.static_eval(expr)`,
  `ct.static_iter(...)` — compile-time checks/unrolling.
- `ct.print(*args, sep=' ', end='\n')`, `ct.printf(format, *args)`,
  `ct.assert_(...)` — device-side debug (use for diagnostics in generated code).
- `ct.kernel` / `ct.function` decorators; `ct.launch(stream, grid, kernel, args)`.
- `ct.Constant[int]` annotation marks a compile-time-constant kernel parameter
  (constant embedding → one compiled variant per value).

---

## 12. Doc vs install mismatches / cautions

- The **public docs page only shows one-line op summaries** (JS-rendered API
  pages); always trust the installed 1.4.0 docstrings (mined above) for exact
  signatures.
- **`ct.iota` does not exist** — it's `ct.arange(size, dtype=...)` (1-D only,
  `dtype` is keyword-only).
- **`ct.select` does not exist** — it's `ct.where`.
- **`ct.dot` does not exist** — 1-D `ct.matmul`/`@` is the dot product.
- `gather`/`scatter` **negative indices are OOB, not Python-negative** — easy
  source of silent wrong results if codegen emits `-1`.
- `ct.mma` does **not** promote operand dtypes (unlike `ct.matmul`); the acc
  dtype dictates output dtype.
- `ct.cat` requires **equal-shape** inputs (power-of-2 rule) and takes a pair.
- `argmax`/`argmin` reject a **tuple** axis.
- `load`'s default `padding_mode=UNDETERMINED` ⇒ OOB lanes are garbage; codegen
  must set an explicit `padding_mode` whenever a tile can straddle the boundary.

---

## 13. Robustness checklist for the DaCe `ct` expansions

1. **Boundary/remainder** → use `load(..., padding_mode=ZERO/...)` + rely on
   `store` dropping partial-OOB writes; stop synthesizing mask tiles for OOB.
2. **Transpose/permute** → push into `load`/`store` `order=` instead of a
   separate `ct.transpose`/`ct.permute` tile op when possible.
3. **Gather/scatter** → keep `check_bounds=True`, pass `mask`/`padding_value`;
   for accumulating scatters emit `ct.atomic_add` (race-free) not `ct.scatter`.
4. **Reductions** → use built-ins with `axis` tuple + `keepdims`; reserve
   `ct.reduce`/`ct.scan` (with tuple payloads) for custom WCR.
5. **MMA** → choose `acc` dtype from the input→acc table; never assume promotion
   in `ct.mma`; realize transposed operands via `order=` on the load.
6. **Casting** → `astype` (value) vs `bitcast` (raw, equal bitwidth); insert
   explicit casts to match the CPU reference's promotion, and avoid unsupported
   mixed-int64 promotions.
7. **Numeric parity** → thread `rounding_mode`/`flush_to_zero` through
   add/sub/mul/div/sum/prod/exp/sqrt when matching a strict reference.

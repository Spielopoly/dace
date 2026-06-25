# cuTile Expansion — Current State Survey

> Read-only survey of what each tileops node's **cuTile** expansion does *today*,
> what it handles, and where the gaps are. Companion to `PROGRESS.md`. All
> file:line references are against the worktree at
> `/workspace/wt/improve-libnodes/dace` as of 2026-06-24.
>
> Scope: only the `cutile` code path (the `ExpandTile*Cutile` classes that emit
> `import cuda.tile as ct` Python tasklets). The `'pure'` CPP path and the K=1
> ISA intrinsics are out of scope except where they reveal a feature the cuTile
> path is missing.

---

## CROSS-CUTTING

### Shared files in the cuTile path

- **`dace/dace/libraries/tileops/nodes/tile_*.py`** — each node defines an
  `ExpandTile<Name>Cutile(ExpandTransformation)` whose `expansion()` returns a
  `nodes.Tasklet(..., language=dace.dtypes.Language.Python)` with a `ct.*` body.
- **`dace/dace/libraries/tileops/_pure_codegen.py`** — despite the name, hosts
  the cuTile-shared geometry helpers used by load/store/mask_gen:
  `cutile_grid_dim_offset` (line 68), `cutile_tile_dim_bids` (line 96),
  `resolve_gather_deps` (line 213), `validate_packed_layout` (line 318),
  `validate_mask_descriptor_lock` (line 360). (`tile_offset`,
  `offset_via_strides`, `gather_lane_offset` are CPP-only — emit `__l<d>`
  flat-index strings, not used by the cuTile path.)
- **`dace/dace/libraries/tileops/_dispatch.py`** — `select_tile_implementation`
  is **CPU-only**; it never returns `'cutile'` for K>=2 and is not the path that
  stamps cuTile. `CuTileSetImplementations` stamps `implementation="cutile"` /
  `target_isa="CUTILE"` directly before `expand_library_nodes()`.
- **`dace/dace/codegen/py/cutile_target.py`** (`CuTilePythonCodeGen`) — the
  Python/cuTile backend that turns the expanded tasklets into a `cuda.tile`
  kernel. AccessNode-centric: `CuTile_Tile` AccessNodes emit their own
  `ct.load`/`ct.store`/`ct.gather`/`ct.scatter`; MapEntry emits `ct.bid`
  bindings; NestedSDFGs become module-level functions returning tiles
  (`_generate_NestedSDFG`, line 1156; `_emit_nsdfg_function`, line 1213).
- **`dace/dace/codegen/py/framecode.py`** — Python-backend frame codegen
  (kernel launch, host arrays, copy states).

### Global limitations (apply to every node)

- **`apply_gpu_transformations()` stamps `GPU_Device` on ALL maps** — a mixed
  SDFG with host-only tasklets fails in the Python backend (per `CLAUDE.md`).
- **Scalars staged as constants** — the cuTile runtime cannot resolve runtime
  scalar arguments; they are baked as constants. Symbol/scalar inputs to tile
  ops therefore inline only via `symstr`, not as live kernel args.
- **Out-of-bounds memlets for non-divisible concrete sizes** — `ct.store`'s
  aligned path and `apply_gpu_transformations()` copy states can address past
  the array end when a concrete dim is not a multiple of the tile width.
- **CuTile power-of-2 constraint** — every `widths` dim must be a power of 2 and
  is a compile-time constant. `widths` length is capped at 1..3 (K in {1,2,3})
  in every node's `__init__`. >3D grids raise `NotImplementedError`
  (`cutile_target.py:1386`).
- **Python tasklets only** — `_generate_Tasklet` raises if the language is not
  Python (`cutile_target.py:998`); node-level instrumentation inside a cuTile
  kernel raises (`cutile_target.py:1005`).
- **`CUTILE_EXPANSION_DESIGN.md` is explicitly marked OUTDATED** (TODO header,
  line 1). Trust the node code, not the doc.
- **Doc-vs-code drift in masked paths** — `tile_ite` computes `_CT_HAS_WHERE`
  (capability probe, line 43-47) and `tile_reduce`'s docstring promises an
  "arithmetic-blend fallback when `ct.where` is absent", but **neither cuTile
  expansion actually branches on it** — both unconditionally emit `ct.where`.
  Dead capability probe + misleading docstrings (see per-node notes).

### cuTile primitive coverage today

| Primitive          | Emitted by                                   |
|--------------------|----------------------------------------------|
| `ct.load`          | TileLoad (aligned), TileStore/Load scalar    |
| `ct.gather`        | TileLoad (strided/replicate/gather)          |
| `ct.store`         | TileStore (aligned)                          |
| `ct.scatter`       | TileStore (masked/strided/gather)            |
| `ct.mma`           | TileMMA                                       |
| `ct.sum/prod/min/max` | TileReduce                               |
| `ct.where`         | TileITE, TileReduce(masked), Binop/Unop(mask)|
| `ct.broadcast_to`  | TileLoad/Store scalar+symbol, MaskGen, Iota   |
| `ct.arange`        | TileIota, TileMaskGen, Load/Store strided    |
| `ct.permute`       | TileLoad/Store transposed axes               |
| `ct.reshape`       | TileLoad (index-tile rank fixup)             |
| `ct.minimum/maximum/abs/exp/...` | TileBinop, TileUnop            |
| `ct.bid`           | every load/store/mask (PID bindings)          |

---

## tile_load — `ExpandTileLoadCutile` (`tile_load.py:192-425`)

### 1. Node properties
`widths` (K=1..3), `target_isa`, `dim_strides` (per-tile-dim affine coeff,
symbolic-capable), `src_dims` (per-tile-dim → source-array dim mapping; empty =
last K dims; enables transpose/permute), `has_mask`, `pad_mode`
(ZERO/NAN/POS_INF/NEG_INF/NEG_ZERO/UNDETERMINED → `ct.PaddingMode`), `src_kind`
(Tile/Scalar/Symbol), `src_expr`, `replicate_factor_per_dim` (group-broadcast,
symbolic-capable), `gather_dims` (sorted SOURCE dims that gather, each with an
`_idx_<d>` connector). The most feature-rich node.

### 2. cuTile expansion behavior
Dispatch in `expansion()`:
- `src_kind="Scalar"` (line 229): `ct.load(...).item()` for a len-1 global array
  else `_src.item()`, wrapped in `ct.broadcast_to(ref, widths)`.
- `src_kind="Symbol"` (line 241): `ct.broadcast_to(symstr(src_expr), widths)`.
- `src_kind="Tile"` (line 243) — four sub-paths:
  - **gather path** (`gather_set` non-empty, line 287): builds one index entry
    per source dim — `_idx_{k}` for gather dims, `ct.arange(...)`-based for
    tile-mapped dims, memlet-begin scalar for unused/stride-0 dims — then
    `ct.gather(_src, (idx...), padding_value=..., mask=_mask?)` (line 323).
  - **aligned path** (default coeffs, no replicate, line 324):
    `ct.load(_src, index=(__pid.../base), shape=(W.../1), padding_mode=...)`.
  - **general strided/replicate path** (line 339): per-source-dim `ct.arange`
    index tiles + `ct.gather(..., padding_value=...)`.
- **Transpose**: when `all_dimensions` is not sorted (line 372),
  `ct.permute(src_code, axes=permute_order)`.
- **Mask** on non-gather paths (line 379): `ct.where(_mask, src_code, pad_value)`.
- **Index-tile rank fixup** (line 394-406): `ct.reshape` when the declared
  `_dst` output shape rank differs from K (the `ONE`-padded index-tile case).
- PIDs resolved via `cutile_tile_dim_bids` (line 267) — per-dim grid axis, so a
  gather index tile walking a non-innermost loop reads the right `ct.bid`.

### 3. Cases handled today
Contiguous aligned block load; strided (non-unit `dim_strides`); group-broadcast
replication; transposed/permuted axes (`src_dims` + `ct.permute`); indirect
gather (per-dim `_idx`); scalar/symbol broadcast; per-lane mask (gather via
`mask=`, others via `ct.where`); multi-dim K=2,3; OOB padding via `pad_mode`;
non-tile outer source dims pinned to their memlet-begin index.

### 4. Gaps / NOT handled
- **`pad_mode` has no `1`/`prod` identity** (`_PAD_MODE_CUTILE`, line 25;
  comment lines 19-32). A load feeding a `prod` reduction cannot install the
  multiplicative identity at load time — routed to the reduction's pre-select.
- **`mask=` on `ct.gather` is "strongly indicated but NOT confirmed"** in the
  cuTile API (design doc L-gs-mask). The masked gather path (line 322) assumes
  it exists.
- **Non-gather masked load uses `ct.where(_mask, ..., pad_value)`** (line 379) —
  a post-load blend, so OOB lanes still read through `ct.load` first; relies on
  `pad_mode` being a safe value for the read.
- **Packed-layout-only** (`validate_packed_layout`, called at `validate`): a
  non-packed-C / non-packed-Fortran `_src` stride pattern raises
  `NotImplementedError` (`_pure_codegen.py:353`). Padded/strided-view sources
  unsupported.
- **`ct.gather` index dtype int32 cast** — strided/replicate paths emit
  `ct.arange(..., dtype=ct.int32)` (lines 308,360); large indices that overflow
  int32 (huge arrays) would silently wrap. Gather `_idx` dtype is validated to
  int32/int64 at `validate` but the arange contributions are hardwired int32.
- **`raise ValueError`**: unrecognized `pad_mode` (line 225), `dim_strides`
  length mismatch (line 277), unrecognized `src_kind` (line 409).
- Subset-begin extraction is best-effort: a non-Range subset falls back to `0`
  for unused dims (line 252-260), which can pin the wrong slice.

### 5. Existing test coverage
See the test-coverage section at the end. (Heavily exercised by orchestrator +
cuTile integration suites; structure asserts on `ct.load`/`ct.gather`.)

---

## tile_store — `ExpandTileStoreCutile` (`tile_store.py:118-357`)

### 1. Node properties
`widths`, `target_isa`, `dim_strides`, `dst_dims` (transpose mapping), `has_mask`,
`src_kind` (Tile/Scalar/Symbol), `src_expr`, **`wcr`** (write-conflict-resolution
lambda — required for collapse/broadcast writes), `gather_dims` (sorted DEST dims
that scatter, each with `_idx_<d>`). Symmetric to TileLoad plus WCR and the
full-tile-write contract.

### 2. cuTile expansion behavior
Three paths after resolving the tile expression per `src_kind` (Tile = `_src`;
Scalar = `ct.load(...).item()`/`.item()` + `ct.broadcast_to`; Symbol =
`ct.broadcast_to(symstr(...))`):
- **Tile-fill** (`is_tile_fill`, line 204): transient dest whose shape == widths
  → plain `_dst = tile_expr` (SSA); masked → `ct.where(_mask, tile_expr, 0)`.
- **Gather-dims scatter** (line 222): `ct.scatter(_dst, (idx...), tile, mask=?)`
  with `_idx_{d}` for scattered dims, `ct.arange`-based for structured dims.
- **Structured store** (line 272):
  - aligned (`is_default_coeffs and not has_mask`, line 293):
    `ct.store(_dst, index=(__pid.../begin), tile=...)` with singleton-axis
    insertion (line 298) + `ct.permute` for transpose (line 301).
  - else (masked or strided, line 314): per-dim `ct.arange` index tiles +
    `ct.scatter(_dst, (idx...), tile, mask=?)`.

### 3. Cases handled today
Aligned contiguous store; strided; transposed (`dst_dims` + `ct.permute`);
scatter via `_idx`; per-lane masked store (always via `ct.scatter`, since
`ct.store` has no mask); scalar/symbol broadcast fill into a register tile;
multi-dim; stride-0 broadcast on scatter dims.

### 4. Gaps / NOT handled
- **WCR / atomic scatter NOT supported** — `node.wcr is not None` raises
  `NotImplementedError` (`tile_store.py:152-156`). Any reduction/accumulating
  store to global memory (the collapse-out / broadcast-write case) is blocked on
  the cuTile path; must use pure CPP. This is the single biggest store gap.
- **Non-full-tile structured stores raise `NotImplementedError`** — the dest
  memlet subset size must equal `widths` under `dst_dims`; partial-tile /
  single-element / scalar→global writes are refused at `validate`
  (`tile_store.py:635-642`). (Skipped for scatter mode.)
- **Aligned `ct.store` can write OOB** for non-divisible concrete sizes (no
  bounds/padding on the store side; the global limitation above).
- **Packed-layout-only** (`validate_packed_layout` on `_dst`,
  `tile_store.py:568`).
- **`ct.scatter` `mask=`** — same "indicated but not confirmed" caveat as gather.
- **`raise ValueError`**: unrecognized `src_kind` (line 197), `dim_strides`
  length mismatch (line 283).

### 5. Existing test coverage
See final section.

---

## tile_mma — `ExpandTileMMACutile` (`tile_mma.py:108-160`)

### 1. Node properties
`widths` = `[M, K_inner, N]` (exactly 3, all positive; cuTile needs powers of 2),
`alpha` (scalar prefactor on A@B), `beta` (scalar prefactor on C; 0=overwrite,
non-zero=accumulate), `target_isa` (informational only — only `pure`/`cutile`
exist). Connectors `_a (M,K_inner)`, `_b (K_inner,N)`, `_cin (M,N)` when beta!=0,
`_c (M,N)`.

### 2. cuTile expansion behavior
Five alpha/beta specializations (line 143-152):
`alpha=1,beta=0` → `ct.mma(_a,_b)`; `alpha=1,beta=1` → `ct.mma(_a,_b,_cin)`;
`beta=0` → `alpha * ct.mma(_a,_b)`; `alpha=1` → `ct.mma(_a,_b) + beta*_cin`;
general → `alpha*ct.mma(_a,_b) + beta*_cin`. `validate()` is called first.

### 3. Cases handled today
2D×2D single-tile GEMM with GEMM-style alpha/beta; accumulate (in-place via
`_cin`/`_c` to same AccessNode) and overwrite; uniform-dtype operands.

### 4. Gaps / NOT handled
- **Uniform dtype only** — mismatched `_a`/`_b`/`_c` dtypes raise
  `NotImplementedError` (`tile_mma.py:242-244`). No mixed-precision MMA
  (e.g. fp16 inputs → fp32 accumulator), which is the *primary* hardware MMA use
  case. Big robustness gap.
- **No transpose flags** — operands must be exactly `(M,K_inner)` / `(K_inner,N)`
  (`validate` line 245-249). A transposed operand (`A.T @ B`) must be permuted
  upstream; the node carries no `trans_a`/`trans_b`.
- **Exactly 2D, 3-tuple widths** — no batched MMA, no K>3 / K<3.
- **`alpha`/`beta` inlined as Python literals** (line 148-152) — symbolic
  alpha/beta would inline as a symbol string; correctness with runtime
  prefactors untested.
- No mask support at all.

### 5. Existing test coverage
See final section.

---

## tile_binop — `ExpandTileBinopCutile` (`tile_binop.py:309-363`)

### 1. Node properties
`op` (`+ - * / % < <= > >= == != && || & | ^ min max **`), `widths`, `has_mask`,
`kind_a`/`kind_b` (Tile/Scalar/Symbol; at least one Tile), `expr_a`/`expr_b`
(inline Symbol exprs), `target_isa`.

### 2. cuTile expansion behavior
Operand ref via `_cutile_operand` (line 332): Symbol → `symstr(expr)`, Tile/Scalar
→ bare connector (NumPy-style broadcast). Op rendered from `_CUTE_OP_EXPR`
(line 282): arithmetic/comparison map to Python operators; `&&`/`||` →
`ct.astype(...,ct.bool_) & / |`; `min`/`max` → `ct.minimum`/`ct.maximum`;
`%`/`/`/`**` map directly. Masked: `rhs = ct.where(_mask, rhs_expr, False)` (line 354).

### 3. Cases handled today
Element-wise binary ops on tiles; Tile op Scalar/Symbol broadcast; logical and
bitwise ops; min/max/pow; multi-dim (NumPy-style broadcast). Mask via `ct.where`.

### 4. Gaps / NOT handled
- **Python `/` and `%` semantics differ from C++** for integer division and
  negative operands — explicit `TODO`s at `tile_binop.py:286,288`. The cuTile
  path emits Python `%`/`/` (line 287,291), so a vectorized cuTile result may
  **disagree with the pure CPP reference** on negative-integer modulo/division.
  Correctness hazard for mixed-backend comparison tests.
- **Masked fill value is `False`** (line 354) — `ct.where(_mask, expr, False)`.
  For a non-bool numeric output the inactive lanes become `0`/`False`; if the
  consumer expects the op's identity (e.g. for a later reduction) this is wrong.
  Hardcoded, not op-aware.
- **No dtype promotion in the cuTile body** — the pure path casts operands
  (`_operand_dtype`, `_promotion_ok`), but the cuTile expansion emits bare
  connectors with no `ct.astype`. Relies on cuTile/NumPy implicit promotion;
  narrowing is refused at `validate` (`NotImplementedError`, line 540) but
  widening behavior on-device is untested.
- **No Scalar-output path** in cuTile — the pure path has an `out_is_scalar`
  branch (no lane loop); the cuTile expansion always emits `_c = <expr>`.

### 5. Existing test coverage
See final section.

---

## tile_unop — `ExpandTileUnopCutile` (`tile_unop.py:182-224`)

### 1. Node properties
`op` (`neg not abs exp log sqrt sin cos floor ceil tanh`), `widths`, `has_mask`,
`kind_a` (Tile/Scalar/Symbol), `expr_a`, `target_isa`.

### 2. cuTile expansion behavior
Operand via `_cutile_operand` (Symbol → `symstr`, else connector). Op from
`_CUTE_UNOP_EXPR` (line 55): `neg`→`(-a)`, `not`→`(True ^ ct.astype(a,ct.bool_))`,
others → `ct.abs/exp/log/sqrt/sin/cos/floor/ceil/tanh`. Masked:
`ct.where(_mask, rhs_expr, False)` (line 215).

### 3. Cases handled today
Element-wise unary math/logical on tiles; Scalar/Symbol broadcast; multi-dim;
mask via `ct.where`.

### 4. Gaps / NOT handled
- **Masked fill value is `False`** (line 215) — same hardcoded `0`/`False`
  inactive-lane issue as binop; not op-aware.
- **No `ct.astype` promotion** in the body (same as binop); narrowing refused at
  `validate` (`NotImplementedError`, line 350), widening untested on-device.
- **No Scalar-output path** — always `_c = <expr>` (pure path has the
  `out_is_scalar` no-loop branch).
- Trig/transcendental result parity with the CPP `std::` reference on-device is
  unverified (e.g. `ct.exp` vs `std::exp` rounding).

### 5. Existing test coverage
See final section.

---

## tile_reduce — `ExpandTileReduceCutile` (`tile_reduce.py:225-286`)

### 1. Node properties
`widths`, `op` (`+ * min max`), `axis` (single dim or `None`=full reduction),
`has_mask`, `target_isa`. Connectors `_src`, `_mask?`, `_dst` (kept-dim shape or
length-1 scalar).

### 2. cuTile expansion behavior
Masked (line 258): pre-set inactive lanes to the op identity via
`ct.where(_mask, _src, identity)` where identity from `_identity_literal_cutile`
(line 186) — `+`→0, `*`→1, `min`→`+inf`/typed-max, `max`→`-inf`/typed-min,
dtype-aware for float/bool/signed/unsigned int. Then
`ct.sum/prod/min/max(rhs, axis=node.axis)` (line 265-272). `_dst = reduce_expr`.

### 3. Cases handled today
Single-axis and full reduction; `+ * min max`; masked reductions via identity
pre-select (dtype-correct identities for float/int/bool); multi-dim input.

### 4. Gaps / NOT handled
- **Docstring promises an arithmetic-blend fallback "when `ct.where` is known
  absent"** and a `NotImplementedError` for masked min/max in that case
  (docstring line 234-240), **but the code never branches** — it always emits
  `ct.where` (line 261). Dead/misleading doc; if `ct.where` is genuinely absent
  the kernel just fails at runtime.
- **`axis=node.axis` passed raw to `ct.sum`** — the DaCe tile-dim axis is assumed
  to match the cuTile tile axis numbering 1:1. Untested whether cuTile's reduce
  axis convention matches after any upstream `ct.permute`.
- **Full reduction output**: cuTile reduce with `axis=None` returns a scalar/0-d
  tile; how that maps onto the `_dst` length-1 transient AccessNode and the
  cross-tile WCR accumulation (caller's job) on-device is the unverified seam.
- **`raise ValueError`** for unsupported identity types (line 209,221),
  `NotImplementedError` for unknown op (line 223,274).
- No `prod` load-time identity (ties back to TileLoad `pad_mode` gap).

### 5. Existing test coverage
See final section.

---

## tile_ite — `ExpandTileITECutile` (`tile_ite.py:128-182`)

### 1. Node properties
`widths`, `kind_t`/`kind_e`/`kind_mask` (Tile/Scalar/Symbol), `expr_t`/`expr_e`/
`expr_mask`, `target_isa`. Per-lane select `_o = mask ? _t : _e`. Uniform dtype
across `_t`/`_e`/`_o` required.

### 2. cuTile expansion behavior
Each operand via `_cutile_operand` (Symbol → `symstr`, Tile/Scalar → connector).
Body: `_o = ct.where(<mask>, <then>, <else>)` (line 166). `validate()` first.

### 3. Cases handled today
Per-lane blend with any of cond/then/else being Tile/Scalar/Symbol; loop-invariant
symbolic condition (no `_mask` connector); multi-dim.

### 4. Gaps / NOT handled
- **`_CT_HAS_WHERE` capability probe (line 43-47) is dead** — computed but never
  consulted by `ExpandTileITECutile`. The "arithmetic-blend fallback / non-finite
  raise" the module comment (line 38-39) describes does not exist in the cuTile
  expansion. If `ct.where` is absent the kernel fails.
- **Uniform-dtype only** — cross-dtype select (Tile/Scalar arms vs `_o`) raises
  `NotImplementedError` (`tile_ite.py:366`); Symbol arms are exempt (cast inline,
  but only in the *pure* path — the cuTile path does no cast).
- **No explicit bool cast on the condition** — a non-bool `_mask` tile is passed
  straight to `ct.where`; relies on cuTile treating it as truthy.
- Output-kind rule: any Tile input forces tile-shape `_o` (`NotImplementedError`,
  line 357).

### 5. Existing test coverage
See final section.

---

## tile_iota — `ExpandTileIotaCutile` (`tile_iota.py:86-154`)

### 1. Node properties
`widths`, `expr` (per-lane body in `__l0..__l{K-1}` + any `extra_inputs` names),
`extra_inputs` (list of extra connector names, e.g. `_idx`/`_src`), `target_isa`.
No `op`/mask. Used to materialize gather/scatter index tiles and affine fills.

### 2. cuTile expansion behavior
- Degenerate all-widths-1 (line 113): substitute `__l{p}`→`0`, rewrite
  `conn[0]`→`conn`, emit `_dst = <expr>`.
- General (line 129): build `__l{k} = ct.arange(W_k, dtype=ct.int32)` (K=1) or
  `ct.broadcast_to(ct.arange(...)[slice], shape)` (K>=2) per dim, then
  `_dst = <expr>` evaluating the raw Python `expr` over those lane arrays and
  the `extra_inputs` connectors.

### 3. Cases handled today
Affine index fills (`i + __l0`, `i + 2*__l0`); indirect per-lane lookups via
`extra_inputs` (`_idx[__l0]`); multi-dim broadcast lane arrays; single-lane
degenerate case.

### 4. Gaps / NOT handled
- **`expr` is emitted verbatim as Python** — there is no validation that `expr`
  is a legal cuTile expression. A CPP-flavored `expr` (e.g. `std::min`, ternary
  `? :`, integer-cast syntax) emitted by an upstream CPP-oriented emitter would
  be invalid Python and crash codegen. The pure and cuTile paths share the same
  `expr` string but have divergent grammars — **this is the silent-trap gap**.
- **`ct.arange` hardwired `dtype=ct.int32`** (line 134,142) — int64 index tiles /
  large arrays can overflow; the dtype is not taken from the `_dst` descriptor.
- **`extra_inputs` indexing `conn[__l0]`** assumes the connector is a cuTile tile
  indexable that way; mixing a global-array `extra_input` with lane-index
  subscript on-device is unverified.
- **No bounds/OOB handling** in the index expression itself.

### 5. Existing test coverage
See final section.

---

## tile_mask_gen — `ExpandTileMaskGenCutile` (`tile_mask_gen.py:53-106`)

### 1. Node properties
`widths`, `iter_vars` (per-dim surrounding-map iter-var names), `global_ubs`
(per-dim exclusive upper-bound expressions), `target_isa`. No inputs; one output
`_o` (bool tile). Output descriptor locked to
`Array(shape=widths, dtype=bool_, storage=Register/CuTile_Tile, transient=True)`.

### 2. cuTile expansion behavior
PID offset via `cutile_grid_dim_offset` (line 85; assumes the K tiled dims are
the innermost/trailing K grid axes). For each dim:
`__offsets{k} = ct.arange(W_k, dtype=ct.int32)`,
`__mask{k} = __offsets{k} + __pid{k}*W_k < (ub)`. K=1 → `_o = __mask0`; K>=2 →
`_o = &`-conjunction of `ct.broadcast_to(__mask{k}[slice], shape)` (line 98).

### 3. Cases handled today
ANY-dim-OOB conjunction iteration mask; K=1,2,3; symbolic upper bounds (inlined
verbatim); the standard boundary mask the reference cuTile kernels use.

### 4. Gaps / NOT handled
- **`global_ubs` inlined verbatim** — the upper-bound expression strings are
  emitted as-is into Python. A CPP-flavored ub expr would be invalid Python
  (same grammar trap as TileIota `expr`). Symbol resolution depends on the
  surrounding kernel scope binding those names.
- **Only the `<` (less-than upper-bound) form** — no lower-bound / interior /
  arbitrary predicate masks; the mask is hardwired to the OOB-tail conjunction.
- **`cutile_grid_dim_offset` positional assumption** (line 85, helper at
  `_pure_codegen.py:68`): correct only when the K tiled dims are the innermost K.
  A mask whose dims are not the trailing grid axes would bind the wrong `ct.bid`.
- **`ct.arange` int32** (line 88) — same overflow caveat.
- **`iter_vars` is read by the pure path but the cuTile path ignores it** —
  cuTile recomputes the global offset from `__pid{k}*W_k` instead of using the
  passed `iter_vars`. A divergence between the two paths' notion of the tile
  base if the map binding is not exactly `__pid*W`.

### 5. Existing test coverage
See final section.

---

## EXISTING TEST COVERAGE (per node)

> Filled from the test survey of
> `tests/passes/vectorization/lib_nodes/`. A "runtime GPU test" compiles and
> runs through `@dace.program` → vectorize → cuTile expansion → compile → run on
> GPU (`@pytest.mark.gpu`, `BackendLanguage.Python`, cupy) and compares to NumPy.
> A "structure test" only asserts on emitted `ct.*` strings / SDFG shape. A
> "pure test" exercises only the CPP `'pure'` expansion.

### GPU-marked test files (all compile + run on device, compare to NumPy)

- `test_cutile_integration.py` — module-level `pytestmark = pytest.mark.gpu`
  (line 22); ~32 runtime tests (vadd/sub/mul/div, dtypes, aligned/unaligned
  sizes, tile widths, op×size cross) + a `gather` runtime+structure block.
- `test_cutile_nested_sdfg_runtime.py` — module-level `pytestmark` (line 32);
  ~38 runtime tests (negate, add-constant, binop add/mul, conditional branches,
  multi-state, symbolic sizes, dtypes, tile widths).
- `test_vectorize_cutile_orchestrator.py` — class-level `@pytest.mark.gpu` on
  `TestRuntimeVadd`, `TestRuntimeMultiDim` (~13 runtime tests).
- `test_cutile_data_copies.py` — class-level `@pytest.mark.gpu` on
  `TestCuTileDataCopiesRuntime` (~8 runtime tests, NumPy-host-array entry).

Structure tests (assert on emitted `ct.*` strings, no execution) live in
`test_cutile_expansions.py` (69 tests). Pure CPP-only suites:
`test_tile_{load,store,binop,reduce,mask_gen}_pure.py`,
`test_tile_binop_promotion.py`, `test_tileops_dispatch.py`.

### PER-NODE TEST MATRIX

| Node          | cuTile structure test (`test_cutile_expansions.py`) | cuTile GPU runtime test | Notes |
|---------------|----------------------------------------------------|-------------------------|-------|
| tile_load     | YES — `test_tile_load_cutile_*` (ct.load, K2, pos_inf pad, masked ct.where, gather 1d/2d/non-innermost) | **YES** — every vadd/dtype/size test in integration + orchestrator + data_copies (all loads go through it); `test_zekinh_gather_matches_numpy` exercises the gather path | Best-covered node |
| tile_store    | YES — `test_tile_store_cutile_*` (ct.store, masked scatter, K2 scatter, strided, **transposed permute**, scalar/symbol fill) | **YES** — every store in the runtime suites; masked-remainder → `ct.scatter` | **WCR / atomic-scatter path has NO test (it just raises)**; non-full-tile path untested |
| tile_mma      | YES — `test_tile_mma_cutile_*` (5 alpha/beta forms + connectors) | **NO** | **NO runtime test. No pure test either.** Mixed-dtype/transpose gaps unverified |
| tile_binop    | YES — `test_tile_binop_cutile_*` (bare op, symbol inline, ct.minimum, `**`) | **YES** — `test_binop_add`, `test_binop_mul` (nsdfg runtime); all integration vadd/sub/mul/div | Negative-int `%`/`/` parity untested; masked-fill-`False` untested at runtime |
| tile_unop     | **NO** — `TileUnop` is not even imported in `test_cutile_expansions.py` | **YES (partial)** — `test_negate` (`y=-x`) runs the cuTile unop path end-to-end | Only `neg` is runtime-exercised; abs/exp/log/sqrt/trig/floor/ceil/tanh/not have **no runtime test**; **no structure test, no pure test** |
| tile_reduce   | YES — `test_tile_reduce_cutile_*` (sum/prod/min/max, masked ct.where, K2 axis0/axis1) | **NO** — deliberately deferred (orchestrator doc lines 20-24: `np.sum` with `widths=(8,4)` hits a `MarkTileDims` failure) | **NO runtime test** — the masked-identity + full-reduction seams are unverified on device |
| tile_ite      | YES — `test_tile_ite_cutile_*` (all-tile, symbol then/else, all-symbol, mixed, scalar mask, K2) | **YES (indirect)** — `test_conditional_true_branch` / `false_branch` (nsdfg runtime) go through a branching NSDFG | No pure test; cross-dtype/non-bool-cond on device unverified |
| tile_iota     | YES — `test_tile_iota_cutile_*` (12 tests: K1/2/3 affine+strided, degenerate, extra_inputs idx/src) | **NO direct** — exercised indirectly only as gather-index tiles inside `test_zekinh_gather_*` | Iota is emitter-internal; the CPP-vs-Python `expr` grammar trap is untested |
| tile_mask_gen | YES — `test_tile_mask_gen_cutile_*` (1d arange+bid, K2 broadcast `&`) | **YES (indirect)** — every unaligned/non-divisible runtime test (`test_vadd_unaligned`, `test_*_non_divisible`, `test_k2_non_divisible`) builds a boundary mask | Only the `<`-OOB form is covered; non-innermost-dim binding untested |

### Nodes with NO cuTile GPU runtime coverage (highest risk)

1. **tile_mma** — no runtime test at all (no pure test either).
2. **tile_reduce** — runtime deliberately deferred (blocked on a `MarkTileDims`
   bug); only structure + pure tests.
3. **tile_unop** — only `neg` runs end-to-end; the 9 other ops (abs/exp/log/sqrt/
   sin/cos/floor/ceil/tanh/not) have **no test of any kind on the cuTile path**,
   and there is no structure test or pure test file for it.

Indirectly-only covered (no dedicated runtime test): **tile_iota**,
**tile_mask_gen**, **tile_ite** (ride along on gather / remainder / conditional
fixtures).

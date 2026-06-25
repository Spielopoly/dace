# node: tile_binop — running log

**Status:** TESTING-COMPLETE (batch B, GPU runtime + structural pass — coordinator-verified 2026-06-24)
**File:** `dace/dace/libraries/tileops/nodes/tile_binop.py`
**New runtime tests:** `tests/passes/vectorization/lib_nodes/test_tile_binop_cutile_runtime.py`

## Current state & gaps (from CURRENT_STATE.md)
Ops: `+ - * / % < <= > >= == != && || & | ^ min max **`. Operand via `_cutile_operand`
(Symbol→symstr, Tile/Scalar→bare connector, NumPy broadcast). Masked: `ct.where(_mask, rhs, False)`.
Gaps:
- **Python `/` and `%` differ from C++** for integer division / negative operands
  (TODO tile_binop.py:286,288). cuTile path emits Python `%`/`/` → may disagree with the pure
  CPP reference on negative-int modulo/division. Correctness hazard for mixed-backend tests.
- **Masked fill hardcoded `False`** (line 354) — inactive lanes → 0/False, not op-aware
  (wrong if a downstream reduction expects the op identity).
- **No `ct.astype` promotion** in cuTile body (pure path casts via `_operand_dtype`/`_promotion_ok`);
  narrowing refused at validate (line 540), widening on-device untested.
- **No Scalar-output path** in cuTile (pure path has `out_is_scalar` no-loop branch).

## Edge-case matrix to cover
- all ops incl. comparisons (bool out), logical `&&`/`||`, bitwise, min/max, `**`.
- negative-int `/` and `%` → MUST match the pure CPP reference (decide: emit C-style via
  trunc-div / `ct` ops, or document+test Python semantics consistently across both backends).
- dtype promotion: mixed-dtype operands (int+float, narrow+wide) → verify widening on device.
- masked: all-true/all-false/partial; op-aware inactive-lane fill where it matters.
- Tile⊕Scalar, Tile⊕Symbol broadcast; multi-dim K≥2 NumPy-style broadcast; symbolic sizes.

## Plan (orchestrator, approved by coordinator 2026-06-24)
Research RESOLVED the negative-int `/`/`%` "decision" into a NON-ISSUE:
- `%`: both backends use Python floor-modulo (pure = `dace::math::py_mod()`). Agree. No fix.
- `/`: DaCe frontend always promotes int `/` → float64 (operators.py ~214), so int `/` never
  arises from `@dace.program`; float true-division agrees across backends. No practical gap.
- `//`: not in `_SUPPORTED_BINOPS` (out of scope).
Action = document WHY it's safe + defensive guard for hand-built SDFGs.

REAL bug found: cuTile `**` FAILS on int32 operands (`TileInternalError: Missing binary
arithmetic implementation for pow, int`) → must cast operands to float before `**`, cast back if
output int. (A2)

Planned edits (all in `tile_binop.py`, NO shared-file edits):
- A1: call `node.validate(...)` at start of cuTile expansion (catch narrowing early).
- A2: int `**` → float-cast then back.
- A3: masked fill dtype-aware (`0` for numeric, `False` for bool) instead of hardcoded `False`.
- A4: explicit `ct.astype` promotion for Symbol/Scalar operands when dtype differs from tile
  operand (matches pure path `_cast`), via `_cutile_dtypes.dace_dtype_to_cutile_str`.
- A5: defensive guard/auto-cast for int `/` (can't arise normally; safety for hand-built SDFGs).
New test file `test_tile_binop_cutile_runtime.py`: 11 classes (arith/cmp/logical/bitwise/min-max/
masked/symbol/scalar/mixed-dtype/multidim-K2/non-divisible), all `@pytest.mark.gpu`, vs NumPy.

## Changes made (orchestrator, coordinator-verified 2026-06-24)
All in `tile_binop.py` (`ExpandTileBinopCutile.expansion()`), NO shared-file edits:
- A1: `node.validate(...)` at expansion start (catch narrowing early).
- A2: int-`**` → `ct.astype(...,ct.float64)` then cast back if output int (fixes real cuTile runtime
  crash "Missing binary arithmetic implementation for pow, int").
- A3: dtype-aware masked fill (`False` for bool, `0` for numeric) — was hardcoded `False`.
- A4: explicit `ct.astype(sym, ct_dtype)` for Symbol operands (matches pure-path promotion);
  skipped for `&&`/`||` which already cast to `ct.bool_`.
- A5: defensive int-`/` float-cast (shares A2 path); TODOs replaced with why-safe docs.
- import `dace_dtype_to_cutile_str` from `_cutile_dtypes`.

## Tests (commands + results)
- NEW `test_tile_binop_cutile_runtime.py` (1042 lines, 36 GPU tests, 10 categories) + NEW
  `test_tile_binop_cutile.py` (25 structural).
- Coordinator re-ran binop runtime + structural + pure + promotion together →
  **92 passed, 1 skipped (38s)**.
- SKIP: `test_mul_symbol_rhs_free` — cuTile Python backend does not forward FREE SDFG symbols
  (`dace.symbol`) into `ct.program` kernels at runtime (platform limitation; literal symbol exprs work).

## Decisions needed / open issues
- RESOLVED: negative-int `/`/`%` parity = non-issue (documented).
- **Carry to BATCH C**: `tile_unop.py` has the SAME hardcoded-`False` masked-fill bug → apply the
  A3 fix there.
- Minor: GPU integer `pow` truncates for non-power-of-2 bases (test uses base=2); free-symbol
  forwarding is a platform limitation.

## Tests (commands + results)
(none yet)

## Decisions needed / open issues
- **Negative-int `/`/`%` semantics**: should cuTile match CPP (C-truncation) or Python (floor)?
  This is a cross-backend-consistency decision — relay to human if non-trivial.

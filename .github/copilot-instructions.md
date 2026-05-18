# Project Guidelines

## Code Style
- Prefer small, targeted changes that preserve existing public APIs unless the task requires API changes.
- For new Python functions, add type hints and keep imports explicit (no `import *`).
- Run formatting and checks before finalizing changes:
  - `pre-commit run --all-files`
- Use spaces for indentation (4 spaces per level), NEVER tabs.
- Make good variable and function names — avoid abbreviations and single-letter names except for tight loop indices. There is no reason to save a few characters at the cost of readability.
- Use docstrings and comments liberally to explain the purpose and rationale of code, especially for non-obvious logic.
- Also use comments to explain what longer code sections are doing at a high level, even if the code is straightforward. This helps future readers understand the intent without having to parse every line.

## Architecture
- DaCe is organized around the SDFG IR and transformation pipeline.
- Use these directories as boundaries when making changes:
  - Core IR and graph model: [dace/sdfg](../dace/sdfg)
  - Transformations and passes: [dace/transformation](../dace/transformation)
  - Code generation backends: [dace/codegen](../dace/codegen)
  - Frontends: [dace/frontend](../dace/frontend)
  - Library nodes and environments: [dace/libraries](../dace/libraries)
- Prefer following existing local patterns in the touched submodule rather than introducing new cross-cutting abstractions.

## Build and Test
- **IMPORTANT: Never create a new Python environment. Always reuse `/venv/main`.**
- Use the workspace virtual environment for Python commands in terminals:
  - `/venv/main/bin/python`
  - `/venv/main/bin/pip`
- Do NOT use `python -m venv`, `conda create`, or other environment creation tools. Use only `/venv/main`.
- Recommended contributor setup:
  - `/venv/main/bin/pip install -e ".[testing,linting]"`
  - `pre-commit install`
- **CRITICAL: Always use the `-B` flag** when running Python tests (`/venv/main/bin/python -B -m pytest ...`). Without `-B`, stale `.pyc` bytecache can mask errors (e.g., TypeError from changed function signatures). This has caused hours of debugging in the past.
- Default test workflow:
  - `/venv/main/bin/python -B -m pytest tests -m "not gpu and not long"`
- Full Python backend suite:
  - `/venv/main/bin/python -B -m pytest /workspace/dace/tests/codegen/python_backend -q`
- Run focused tests for touched areas (examples):
  - `/venv/main/bin/python -B -m pytest tests/transformations -m "not gpu"`
  - `/venv/main/bin/python -B -m pytest tests/sdfg -m "not gpu"`
- GPU-only changes should also run:
  - `/venv/main/bin/python -B -m pytest tests -m "gpu" --timeout=300`
- cuTile library changes:
  - `/venv/main/bin/python -B tests/cutile/all_cutile_tests.py` (runs full cuTile suite)
  - `/venv/main/bin/python -B -m pytest tests/cutile/cutile_test.py -x` (core unit tests)
  - `/venv/main/bin/python -B -m pytest tests/cutile/cutile_frontend_test.py -x` (pipeline integration tests)
- For unknown reason you cannot import the dace module in interactive python sessions, so create a temporary test script that imports the module and run it with `/venv/main/bin/python` if you need to quickly test something interactively.

## Conventions
- Place tests under [tests](../tests) with names matching `test_*.py`, `*_test.py`, or `*_cudatest.py` (see [pytest.ini](../pytest.ini)).
- Use pytest markers for hardware/software requirements instead of ad-hoc skips (see [pytest.ini](../pytest.ini)).
- Keep environment-dependent behavior explicit in tests (cache mode, config toggles) when reproducing codegen/serialization behavior.
- In the Python backend path, default map schedules are normalized to `Sequential` before backend dispatch, so schedule-sensitive fixes often need to be reasoned about before target-specific code generation.
- For cuTile pipeline work, ensure regressions cover unnecessary data movement removal before scalar-to-tile lowering.
- Prefer detailed comments and documentation — don't shorten docstrings or inline comments during refactoring.
- Don't create local variables for `self.xxx` unless the value is used many times or the expression is long. Access through `self.` directly.
- Don't use hardcoded connector name strings (e.g. `"_a"`, `"_b"`) when the name is available in a data structure like `LibraryNodeInfo`. Look up connector names from the registry/info objects instead.
- When merging similar classes, prefer a single unified class over backward-compatible aliases. Replace all usages rather than maintaining aliases.

## Code Quality
- After making code changes, **always run the `code-quality-reviewer` agent** as a subagent on the changed files before considering the task complete. Address any findings before finishing.
- Every function, class, and module should have a docstring that explains **what** it does and **why** it exists.
- All function parameters and return values should have type hints. Omit return type only when the function returns `None` or when the return type is obvious (e.g. `__init__`).
- Prefer reusing existing DaCe utilities (e.g. `SDFGState.remove_memlet_path`, `sdfg.utils.*`, `subsets.*`) over reimplementing graph manipulation logic. Search the codebase before writing new utility code.
- Keep functions small and single-responsibility. If a function does multiple logically distinct steps, split it.
- Use descriptive variable and function names — AVOID ABBREVIATIONS or single-letter names outside tight loop indices.
- Avoid `sp.simplify()` for equality/zero checks on symbolic expressions — it is expensive and unreliable. Prefer structural comparison (`==`, `!=`) or `.is_zero` where appropriate.
- Flag and remove dead code, unused imports, and stale comments during any refactoring pass.

## cuTile Library (`dace/libraries/cutile/`)
- **Library nodes** (`nodes/`):
  - `TileOpLibraryNode` (`nodes/op.py`) — unified element-wise op node for both binary and unary operations. Properties: `op`, `constant1`, `constant2`, `tile_shape`. Connectors: `_a` (optional input), `_b` (optional input, binary only), `_c` (output).
  - `TileRuntimeMaskedOpLibraryNode` (`nodes/op_runtime_map.py`) — masked variant with additional `_m` (mask) and optional `_c_in` connectors. Imports `_op_cpp_expr`, `_BINARY_OPS`, `_UNARY_OPS`, `_ALL_OPS` from `op.py`.
  - Both support `constant1`/`constant2` to replace array inputs with compile-time constants.
- **Op registry** (`op_registry.py`): maps `(operator, TaskletType, MaskType)` triples to `LibraryNodeInfo` via `register_op()`. `LibraryNodeInfo` stores connector names (`rhs1`, `rhs2`, `out`, `mask_in`, `out_in`) and constant values. `match_tasklet_to_tile_library_node()` uses `classify_tasklet` from `dace.sdfg.tasklet_utils`.
- **DaCe library registration**: `register_library()` iterates `module.__dict__` for `LibraryNode` subclasses. Never put multiple names for the same class in a module's namespace or `__all__` — this causes double-registration errors.
- **Transformation architecture**: `_ScalarToTileBase` (template method base) with two concrete child classes:
  - `ScalarToTileCanonical` — 0-based, unit-stride inner maps → unmasked tile library nodes
  - `ScalarToTileMasked` — non-canonical inner maps → runtime-masked tile library nodes with preload
- **IfElseMapToTileWhere** (`if_else_to_where_select.py`): Converts if-else patterns inside maps to `TileIfElseOpLibraryNode` select operations. Key internals:
  - `_get_branch_output_array()` chain-walks AccessNodes from tasklet to final sink (uses `visited` set to avoid cycles)
  - `_fix_staging_memlets` adjusts memlets for staging arrays — note: do not add dimensionality filtering to `_needs_fix()` (previously caused SIGABRT by skipping 1D arrays)
- **PatternNode bug**: `PatternNode.__get__` resolves nodes by integer index in the state's node list. Adding/removing graph nodes shifts indices, causing descriptors to return wrong nodes. Always capture actual node object references at the start of `apply()` (before any graph modifications) and use those throughout. Never use PatternNode descriptors (`self.outer_map_entry` etc.) after modifying the graph.
- **Pipeline** (`pipeline.py`): `CuTilePipeline` is a 10-step transformation pipeline:
  1. **Simplify** — trivial tasklet elimination + standard simplify
  2. **LoopToMap** — convert state-machine loops into maps
  3. **MapCollapse** — collapse nested maps (optional)
  3b. **MapTiling + IfElseMapToTileWhere** — tile maps containing ConditionalBlock NestedSDFGs
  4. **SplitTasklets** — split multi-statement tasklets into single ops
  5. **MapFission + MapCollapse** — fission maps into single-op maps; merge directly-nested 1D maps
  6. **Simplify** — clean up after preprocessing
  7. **MapTiling** — tile maps to given tile shape (optional)
  8. **Normalize ConditionalBlock** — normalize conditional blocks in NestedSDFGs
  8b. **IfElseMapToTileWhere** — handle patterns not caught in step 3b
  9. **ScalarToTile** — replace scalar tasklets with cuTile library nodes (`ScalarToTileCanonical`, `ScalarToTileMasked`)
  10. **Simplify** — final cleanup
- **Tests**: `tests/cutile/*.py` Run all tests after any cuTile changes to check for regressions.

## Memlet Propagation (`dace/sdfg/propagation.py`)
- The propagation pattern dispatch chain is: `AffineSMemlet` → `ModuloSMemlet` → `ConstantSMemlet` → `GenericSMemlet`. Each pattern's `can_be_applied()` filters what it handles; rejected cases fall through.
- **AffineSMemlet stride shortcut**: The `i:i+stride` shortcut (returns stride=1) is only valid when `multiplier == 1`. For non-identity access like `A[2*i]`, removing this guard silently loses stride information.
- **GenericSMemlet stride**: Computes `skip = |multiplier| * map_stride` for single-parameter affine access. Multi-parameter expressions (e.g., `A[2*i + 3*j]`) fall back to `stride=1` (safe overapproximation).
- **`propagate_subset` defined_variables**: Uses `_freesyms()` (not `symlist()`) to detect symbols in ranges — this correctly handles symbols nested inside `int_floor()`, `ceiling()`, etc.

## MapFission (`dace/transformation/dataflow/map_fission.py`)
- In `can_be_applied()`, use `nsdfg.sdfg.edges()` (top-level only) to check for map parameter references in interstate edges. Do NOT use `all_interstate_edges()` — it recurses into nested control flow regions (e.g., `LoopRegion`, `ConditionalBlock`) and causes false rejections when inner loops reuse map parameter names.
- When calling `propagate_subset` for write-back edges (`_is_data_src is False`), pass `use_dst=True` to get correct direction.
- Only call `propagate_memlets_state` for `expr_index == 0` to avoid corrupting inner NSDFG edges.

# Known Issues
- Cannot use `from __future__ import annotations` because it messes up type hints from dace. But the python version is new enough that it doesn't matter and we can use type hints anyway. Just don't add the future import to any files.
- **SIGABRT in compiled code cannot be caught by pytest `xfail`** — use `@pytest.mark.skip(reason="...")` instead. This affects symbolic masked strided patterns that crash at the code generation level.
- **Per-test `@pytest.mark.filterwarnings`** is preferred over blanket `pytest.ini` warning filters for traceability. Use when warnings are expected for specific test configurations (e.g., `validate_subsets` with symbolic ranges).
- **cuTile masked scalar-to-tile descriptor symbols**: In masked scalar-to-tile paths, avoid leakage of map-local parameters into transient descriptor expressions (e.g., `tile_i`, `tile_j` in shape/stride formulas). Subsequent free-symbol analysis may then classify these leaked symbols as required program arguments, leading to `KeyError: Missing program argument "tile_i"` at call time.

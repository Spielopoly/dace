# Project Guidelines

## Code Style
- Follow the style and contribution rules in [CONTRIBUTING.md](../CONTRIBUTING.md).
- Prefer small, targeted changes that preserve existing public APIs unless the task requires API changes.
- For new Python functions, add type hints and keep imports explicit (no `import *`).
- Run formatting and checks before finalizing changes:
  - `pre-commit run --all-files`

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
- Default test workflow:
  - `/venv/main/bin/python -m pytest tests -m "not gpu and not long"`
- Run focused tests for touched areas (examples):
  - `/venv/main/bin/python -m pytest tests/transformations -m "not gpu"`
  - `/venv/main/bin/python -m pytest tests/sdfg -m "not gpu"`
- GPU-only changes should also run:
  - `/venv/main/bin/python -m pytest tests -m "gpu" --timeout=300`
- For unknown reason you cannot import the dace module in interactive python sessions, so create a temporary test script that imports the module and run it with `/venv/main/bin/python` if you need to quickly test something interactively.

## Conventions
- Place tests under [tests](../tests) with names matching `test_*.py`, `*_test.py`, or `*_cudatest.py` (see [pytest.ini](../pytest.ini)).
- Use pytest markers for hardware/software requirements instead of ad-hoc skips (see [pytest.ini](../pytest.ini)).
- Keep environment-dependent behavior explicit in tests (cache mode, config toggles) when reproducing codegen/serialization behavior.
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
- Use descriptive variable and function names — avoid abbreviations or single-letter names outside tight loop indices.
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
- **PatternNode bug**: `PatternNode.__get__` resolves nodes by integer index in the state's node list. Adding/removing graph nodes shifts indices, causing descriptors to return wrong nodes. Always capture actual node object references at the start of `apply()` (before any graph modifications) and use those throughout. Never use PatternNode descriptors (`self.outer_map_entry` etc.) after modifying the graph.
- **Pipeline** (`pipeline.py`): `apply_cutile_pipeline()` runs `TrivialTaskletElimination` + `ScalarToTileCanonical` + `ScalarToTileMasked`.
- **Tests**: `tests/cutile/cutile_test.py` and `tests/cutile/cutile_frontend_test.py`, `tests/cutile/cutile_if_else_op_test.py`, `tests/cutile/cutile_if_else_test.py`. Run all after any cuTile changes:
  - `/venv/main/bin/python -m pytest tests/cutile/cutile_test.py tests/cutile/cutile_frontend_test.py tests/cutile/cutile_if_else_op_test.py tests/cutile/cutile_if_else_test.py -x -q`

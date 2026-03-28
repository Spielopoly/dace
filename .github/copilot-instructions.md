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

## Conventions
- Place tests under [tests](../tests) with names matching `test_*.py`, `*_test.py`, or `*_cudatest.py` (see [pytest.ini](../pytest.ini)).
- Use pytest markers for hardware/software requirements instead of ad-hoc skips (see [pytest.ini](../pytest.ini)).
- Keep environment-dependent behavior explicit in tests (cache mode, config toggles) when reproducing codegen/serialization behavior.
- For cuTile pipeline work, ensure regressions cover unnecessary data movement removal before scalar-to-tile lowering.
- Prefer detailed comments and documentation — don't shorten docstrings or inline comments during refactoring.

## cuTile Library (`dace/libraries/cutile/`)
- **Transformation architecture**: `_ScalarToTileBase` (template method base) with two concrete child classes:
  - `ScalarToTileCanonical` — 0-based, unit-stride inner maps → unmasked tile library nodes
  - `ScalarToTileMasked` — non-canonical inner maps → runtime-masked tile library nodes with preload
- **PatternNode bug**: `PatternNode.__get__` resolves nodes by integer index in the state's node list. Adding/removing graph nodes shifts indices, causing descriptors to return wrong nodes. Always capture actual node object references at the start of `apply()` (before any graph modifications) and use those throughout. Never use PatternNode descriptors (`self.outer_map_entry` etc.) after modifying the graph.
- **Op registry** (`op_registry.py`): maps `(operator, TaskletType, MaskType)` triples to `LibraryNodeInfo` via `@register_matcher`. The `match_tasklet_to_tile_library_node()` function uses `classify_tasklet` from `dace.sdfg.tasklet_utils`.
- **Pipeline** (`pipeline.py`): `apply_cutile_pipeline()` runs `TrivialChainElimination` + `ScalarToTileCanonical` + `ScalarToTileMasked`.
- **Tests**: `tests/cutile_test.py` (54 test functions), `tests/cutile_frontend_test.py` (6 test functions). Run both after any cuTile changes.

## Documentation Links
- Project overview and quick start: [README.md](../README.md)
- Contribution and coding rules: [CONTRIBUTING.md](../CONTRIBUTING.md)
- Design documentation index: [doc/design/README.md](../doc/design/README.md)
- Setup and packaging details: [setup.py](../setup.py)
- CI examples for robust test commands:
  - [general-ci.yml](../.github/workflows/general-ci.yml)
  - [gpu-ci.yml](../.github/workflows/gpu-ci.yml)

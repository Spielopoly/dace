## Plan Summary

Rewrite the Python-backend `CopyLibraryNode` expansion to emit copy code directly in the expansion (as bare Tasklets) rather than delegating to the backend's `copy_memory()` dispatcher via a NestedSDFG wrapper. Three expansion classes handle different Python-backend copy scenarios: same-storage/H2D via `_cpy_out = _cpy_in`, D2H via `_cpy_out = _cpy_in.get()`, and the existing single-element `ExpandTasklet`. The `select_copy_implementation` routing is updated to distinguish D2H from other Python-backend copies.

## Requirements

- **Explicit**: `ExpandPython` must directly emit copy code (no NestedSDFG delegation to `copy_memory`)
- **Explicit**: Handle all copy directions: same-storage, H2D (host→device), D2H (device→host)
- **Explicit**: D2H copies must call `.get()` because numpy cannot read cupy arrays
- **Inferred**: H2D copies work via CuPy's `__setitem__` (numpy→cupy is implicit)
- **Inferred**: Existing C++ backend expansions must be untouched
- **Inferred**: `cutile_target.py` and `python_target.py` `copy_memory()` methods remain as fallbacks for raw AN→AN edges not lifted by `InsertExplicitCopies`
- **Inferred**: Existing test structure and runtime parity must be preserved

## Architecture / Approach

**Approach**: Replace the single `ExpandPython` (NestedSDFG with bare AN→AN edge) with two bare-Tasklet expansions:

1. **`ExpandPython`** (REWRITE): Returns `Tasklet(code="_cpy_out = _cpy_in")`. Handles same-storage copies (numpy `dst[:] = src[:]`) and H2D copies (CuPy `gpu[:] = numpy[:]` — CuPy's `__setitem__` handles the transfer implicitly).

2. **`ExpandPythonD2H`** (NEW): Returns `Tasklet(code="_cpy_out = _cpy_in.get()")`. Handles D2H copies (GPU→CPU). The `.get()` call converts the cupy array/view to numpy before the output memlet write, which is necessary because numpy's `__setitem__` cannot accept cupy arrays.

**Why bare Tasklets work**: The Python backend's `_generate_Tasklet` (line 646 of `python_target.py`) reads input memlets via `_read_expr()` (producing e.g. `_cpy_in = A[0:N]`), runs the tasklet body, then writes output memlets via `_emit_memlet_write()` (producing e.g. `gpu_A[0:N] = _cpy_out`). The memlet subsetting handles the array slicing; the tasklet body only needs to pass/transform the value.

**Why not a single expansion with `.get()` everywhere**: `.get()` on a numpy array would fail (numpy arrays don't have `.get()`). Splitting D2H into its own expansion keeps the code simple and the routing explicit.

**Alternative rejected**: A single expansion using `isinstance` checks at runtime (e.g., `_cpy_out = _cpy_in.get() if hasattr(_cpy_in, 'get') else _cpy_in`). This adds runtime overhead and obscures the copy direction in the IR.

## Files to Modify

### `dace/libraries/standard/nodes/copy_node.py`
This is the primary file. All changes are here.

- **`select_copy_implementation`** (lines 79–99): Add D2H routing between the single-element Tasklet check and the `'Python'` fallback
- **`ExpandPython`** (lines 794–824): Rewrite to return a bare Tasklet instead of NestedSDFG
- **New `ExpandPythonD2H`** class: Add after `ExpandPython` (after line 824)
- **`CopyLibraryNode.implementations`** dict (line 846–856): Add `'PythonD2H': ExpandPythonD2H`
- **`CopyLibraryNode` docstring** (lines 828–843): Add `'PythonD2H'` to the implementation list

### `dace/transformation/passes/vectorization/vectorize_cutile.py`
- **Docstring update** (lines 50–62, 213–218): Update the description of `ExpandPython` — it no longer lowers to `.set()`/`.get()` via `copy_memory()`. H2D uses implicit CuPy assignment; D2H uses `ExpandPythonD2H` with `.get()`.

### `tests/passes/vectorization/lib_nodes/test_cutile_data_copies.py`
- **`TestCuTileDataCopiesCodegen`** (lines 345–385): Update assertions — H2D no longer generates `.set(`, D2H generates `.get()` instead of `.get(out=`
- **`TestCuTileExplicitCopiesStructure.test_knob_codegen_valid_single_launch`** (line 555–564): Update `.set(` and `.get(out=` assertions
- **`TestCuTileExplicitCopiesStructure.test_knob_2d_codegen_valid`** (lines 566–573): Update `.set(` assertion if present
- **Add new unit tests** for expansion routing and behavior

## Files to Create

None — all changes fit within existing files.

## Implementation Steps

### Step 1: Rewrite `ExpandPython` and add `ExpandPythonD2H`

- **Files:** `dace/libraries/standard/nodes/copy_node.py`
- **Description:** Replace the NestedSDFG expansion with a bare Tasklet; add the D2H variant.
- **Details:**

  **Rewrite `ExpandPython` (replace lines 794–824):**
  ```python
  @library.expansion
  class ExpandPython(ExpandTransformation):
      """Multi-element Python-backend copy: ``_cpy_out = _cpy_in`` as a bare
      Tasklet.  Handles same-storage copies (numpy/CuPy slice assignment) and
      H2D copies (CuPy's ``__setitem__`` accepts numpy arrays implicitly).

      For D2H (GPU -> CPU) use :class:`ExpandPythonD2H`, which calls ``.get()``
      to materialise a numpy array before the output memlet write.
      """
      environments = []

      @staticmethod
      def expansion(node, parent_state, parent_sdfg):
          return nodes.Tasklet(
              node.name,
              inputs={CopyLibraryNode.INPUT_CONNECTOR_NAME: None},
              outputs={CopyLibraryNode.OUTPUT_CONNECTOR_NAME: None},
              code=f"{CopyLibraryNode.OUTPUT_CONNECTOR_NAME} = {CopyLibraryNode.INPUT_CONNECTOR_NAME}",
              language=dace.Language.Python,
          )
  ```

  **Add `ExpandPythonD2H` (new class after `ExpandPython`):**
  ```python
  @library.expansion
  class ExpandPythonD2H(ExpandTransformation):
      """D2H (GPU -> CPU) copy for the Python backend: ``_cpy_out = _cpy_in.get()``.

      Calls CuPy's ``.get()`` to materialise a numpy array from a GPU-resident
      cupy array/view.  Needed because numpy's ``__setitem__`` cannot accept
      cupy arrays directly.
      """
      environments = []

      @staticmethod
      def expansion(node, parent_state, parent_sdfg):
          return nodes.Tasklet(
              node.name,
              inputs={CopyLibraryNode.INPUT_CONNECTOR_NAME: None},
              outputs={CopyLibraryNode.OUTPUT_CONNECTOR_NAME: None},
              code=f"{CopyLibraryNode.OUTPUT_CONNECTOR_NAME} = {CopyLibraryNode.INPUT_CONNECTOR_NAME}.get()",
              language=dace.Language.Python,
          )
  ```

  **Connector type `None`**: The Python backend's `_generate_Tasklet` does not use connector types for code generation — it reads from memlets. `None` means "untyped" and avoids potential type-mismatch issues when carrying multi-element data through a Tasklet connector.

- **Depends on:** None

### Step 2: Update `select_copy_implementation` routing

- **Files:** `dace/libraries/standard/nodes/copy_node.py`
- **Description:** Add D2H detection between the existing Tasklet check and Python fallback.
- **Details:**

  **Replace lines 96–99** with:
  ```python
      if is_python_backend(parent_state):
          if single_elt and not _is_cross_cpu_gpu(inp.storage, out.storage, node, parent_state):
              return 'Tasklet'
          # D2H: GPU-resident source → non-GPU destination requires .get()
          if (inp.storage in dtypes.GPU_RESIDENT_STORAGES
                  and out.storage not in dtypes.GPU_RESIDENT_STORAGES):
              return 'PythonD2H'
          return 'Python'
  ```

  **Routing logic explained:**
  | Scenario | single_elt | cross_cpu_gpu | D2H check | Result |
  |---|---|---|---|---|
  | Single-element same-side | True | False | — | `'Tasklet'` |
  | Single-element H2D | True | True | src not GPU | `'Python'` |
  | Single-element D2H | True | True | src GPU, dst not GPU | `'PythonD2H'` |
  | Multi-element same-side | False | — | src not GPU | `'Python'` |
  | Multi-element H2D | False | — | src not GPU | `'Python'` |
  | Multi-element D2H | False | — | src GPU, dst not GPU | `'PythonD2H'` |

- **Depends on:** Step 1

### Step 3: Update `CopyLibraryNode.implementations` dict and docstring

- **Files:** `dace/libraries/standard/nodes/copy_node.py`
- **Description:** Register `ExpandPythonD2H` in the implementations dict and update the class docstring.
- **Details:**

  **Add to `implementations` dict (after line 855):**
  ```python
  "PythonD2H": ExpandPythonD2H,
  ```

  **Update `CopyLibraryNode` docstring** (lines 828–843): Add `PythonD2H` to the implementation list description. Change the sentence about `Python` to mention that it handles same-storage and H2D, while `PythonD2H` handles D2H with `.get()`.

- **Depends on:** Step 1

### Step 4: Update `vectorize_cutile.py` docstrings

- **Files:** `dace/transformation/passes/vectorization/vectorize_cutile.py`
- **Description:** Update the docstrings that describe how `ExpandPython` lowers copies.
- **Details:**

  **Lines 50–62** (class docstring): Update the description of step 6b. Change:
  > "through a library node whose `ExpandPython` expansion lowers back to the identical `.set()` / `.get()`"

  To something like:
  > "through library nodes whose `ExpandPython` / `ExpandPythonD2H` expansions emit bare Tasklets (`_cpy_out = _cpy_in` for same-storage / H2D, `_cpy_out = _cpy_in.get()` for D2H)"

  **Lines 213–218** (inline comment): Update similarly. Change:
  > "Their ExpandPython expansion (resolved at codegen, where the backend is already Python) lowers back to the same .set()/.get()."

  To something like:
  > "Their ExpandPython / ExpandPythonD2H expansions (resolved at codegen, where the backend is already Python) emit bare Tasklets: `_cpy_out = _cpy_in` for H2D (CuPy handles the transfer implicitly) and `_cpy_out = _cpy_in.get()` for D2H."

- **Depends on:** Steps 1–3

### Step 5: Update existing codegen tests

- **Files:** `tests/passes/vectorization/lib_nodes/test_cutile_data_copies.py`
- **Description:** Update assertions in codegen tests to reflect the new expansion behavior.
- **Details:**

  **`test_codegen_contains_set` (line 348–353):** H2D copies no longer generate `.set(`. The expansion produces a Tasklet `_cpy_out = _cpy_in`, and the Python backend emits `gpu_arr[subset] = _cpy_in`. Rename the test to `test_codegen_h2d_tasklet` (or similar) and assert that the generated code does NOT contain `.set(` (confirming the expansion-based path) or assert it contains `_cpy_out = _cpy_in` or similar marker. Alternative: just check that the code is valid Python and the runtime tests pass.

  **`test_codegen_contains_get_out` (line 355–360):** D2H copies now generate `.get()` (not `.get(out=...)`). Update assertion from `assert ".get(out=" in code` to `assert ".get()" in code`.

  **`test_knob_codegen_valid_single_launch` (line 555–564):** Update:
  - Remove `assert ".set(" in code`
  - Change `assert ".get(out=" in code` to `assert ".get()" in code`

  **`test_knob_2d_codegen_valid` (line 566–573):** Update:
  - Remove `assert ".set(" in code`
  - Add `assert ".get()" in code` if D2H is expected

- **Depends on:** Steps 1–3

### Step 6: Add new tests for expansion routing and behavior

- **Files:** `tests/passes/vectorization/lib_nodes/test_cutile_data_copies.py`
- **Description:** Add unit tests verifying the expansion routing and the generated code patterns.
- **Details:**

  **New test class `TestCopyExpansionRouting`** (no GPU required):

  1. **`test_select_h2d_routes_to_python`**: Build an SDFG with a `CopyLibraryNode` between `CPU_Heap` and `GPU_Global` arrays. Verify `select_copy_implementation` returns `'Python'`.

  2. **`test_select_d2h_routes_to_python_d2h`**: Build an SDFG with a `CopyLibraryNode` between `GPU_Global` and `CPU_Heap` arrays. Verify `select_copy_implementation` returns `'PythonD2H'`.

  3. **`test_select_same_storage_routes_to_python`**: Build an SDFG with same-storage (CPU_Heap → CPU_Heap). Verify routing is `'Python'`.

  4. **`test_select_single_element_same_side_routes_to_tasklet`**: Single-element same-storage. Verify routing is `'Tasklet'`.

  5. **`test_expand_python_returns_tasklet`**: After applying `VectorizeCuTile`, expand library nodes, check that H2D copies produce Tasklet nodes (not NestedSDFG).

  6. **`test_expand_d2h_returns_tasklet_with_get`**: After expansion, check D2H copies produce Tasklet nodes with `.get()` in the code.

  7. **`test_d2h_codegen_contains_get`**: Generate code and verify `.get()` appears.

  8. **`test_h2d_codegen_no_set`**: Generate code and verify `.set(` does NOT appear (i.e., H2D is handled via regular assignment).

  **Runtime test additions** (in existing `TestCuTileExplicitCopiesRuntime`, marked `@pytest.mark.gpu`):

  9. **Existing tests should continue to pass** — they test end-to-end correctness which is unaffected by the internal expansion mechanism change.

- **Depends on:** Steps 1–5

## Parallelization

- **Independent workstreams:**
  - Steps 1–3 are tightly coupled (all in `copy_node.py`) and should be done as one unit.
  - Step 4 (docstring update in `vectorize_cutile.py`) is independent but trivial.
  - Steps 5–6 (test updates) depend on Steps 1–3 but are independent of Step 4.

- **File assignments per workstream:**
  - Workstream A (implementation): `dace/libraries/standard/nodes/copy_node.py`
  - Workstream B (docstring): `dace/transformation/passes/vectorization/vectorize_cutile.py`
  - Workstream C (tests): `tests/passes/vectorization/lib_nodes/test_cutile_data_copies.py`

- **Sequential dependencies:** A must complete before C (tests depend on the new code).

## Test Strategy

### Existing Tests to Run

- `tests/passes/vectorization/lib_nodes/test_cutile_data_copies.py`: Primary test file. Structure tests should pass unchanged. Codegen tests need assertion updates. Runtime tests (GPU) should pass unchanged.
- `tests/passes/vectorization/` (full suite): Broader vectorization regression check.
- `tests/codegen/python_backend/`: Python backend regression check.

### New Tests to Write

- In `tests/passes/vectorization/lib_nodes/test_cutile_data_copies.py`:
  - Routing tests: verify `select_copy_implementation` returns correct implementation for H2D, D2H, same-storage, single-element
  - Expansion structure tests: verify the expansion returns bare Tasklets (not NestedSDFGs)
  - Codegen pattern tests: verify generated code contains/does-not-contain expected patterns

### Tests That May Need Updates

- `test_cutile_data_copies.py`:
  - `test_codegen_contains_set`: assertion change (`.set(` → remove or invert)
  - `test_codegen_contains_get_out`: assertion change (`.get(out=` → `.get()`)
  - `test_knob_codegen_valid_single_launch`: assertion change (remove `.set(`, update `.get(out=` → `.get()`)
  - `test_knob_2d_codegen_valid`: assertion change (remove `.set(`, add `.get()` if needed)

## Risks and Considerations

1. **CuPy implicit H2D**: The H2D path relies on CuPy's `__setitem__` accepting numpy arrays. This is well-established CuPy behavior and is already how the cuTile pipeline works when `insert_explicit_copies=False` (the `copy_memory` path in `python_target.py` does `dst[:] = src[:]` for same-storage, which also works for cross-storage via CuPy). Risk: low.

2. **Connector type `None`**: Using `None` for Tasklet connectors is standard practice (e.g., many tileops expansions do this). The Python backend does not use connector types for codegen. Risk: negligible.

3. **Register storage edge case**: The D2H routing check (`inp.storage in GPU_RESIDENT_STORAGES and out.storage not in GPU_RESIDENT_STORAGES`) does not account for `Register` storage inside GPU scope. However, in the Python backend/cuTile pipeline, copies between Register and CPU_Heap are not a practical scenario — `apply_gpu_transformations()` only creates `CPU_Heap <-> GPU_Global` copies. Risk: negligible for current usage; can be extended if needed.

4. **Backward compatibility**: The `cutile_target.py` and `python_target.py` `copy_memory()` methods remain as fallbacks for raw AN→AN edges (when `insert_explicit_copies=False`). The `.set()`/`.get(out=...)` path is preserved for that case. Risk: none.

5. **NestedSDFG overhead removed**: The old `ExpandPython` created a NestedSDFG wrapper, which has codegen overhead (function definition/call, argument binding). The bare Tasklet is lighter and avoids the NSDFG scalar-bridge machinery. This is a net improvement. Risk: none.

## Open Questions

1. **Should `ExpandPython` validate storage compatibility?** Currently it accepts anything — the routing in `select_copy_implementation` ensures it's only called for same-storage or H2D. Adding a validation guard is optional but defensive. Recommendation: skip for now — the routing is the single entry point and is well-tested.

2. **Should the `_emit_memlet_write` call in `_generate_Tasklet` use `numpy.copyto` for full-array writes?** Looking at line 440–441 of `python_target.py`, `_emit_memlet_write` uses `numpy.copyto(target, value)` when the target is an Array with no subset. This might interact with H2D copies (would call `numpy.copyto(gpu_array, numpy_array)` which numpy can't handle for cupy targets). However, the copy edges from `apply_gpu_transformations()` always carry subsets (full-range like `0:N`), so this path should not be hit. If it is, the `target[subset] = value` fallback (line 443) handles it correctly.

# cuTile all-fixes and `extended` merge session report

Date: 2026-08-05
Repository: `dace`
Source branch: `agent/cutile-all-fixes`
Published branch: `codex/cutile-all-fixes-extended-merge`
Fetched upstream tip: `upstream/extended` at `f696d113a8bbe1422fce006ee8c32feccb9b5123`

## Outcome

The all-fixes branch was cleaned of two fixes that were superseded by newer
implementations on `upstream/extended`, tested to establish a pre-merge
baseline, and merged with the fetched upstream branch. The completed history
contains:

- `0745543c6` — `Drop fixes superseded by extended`
- `f78656774` — `Merge remote-tracking branch 'upstream/extended' into agent/cutile-all-fixes`

The merge retains the cuTile/Python-backend work from all-fixes while adopting
the newer canonicalization, code-generation, CUDA-warp, and tile-operation
changes from `extended`. Focused CPU and GPU integration coverage passes. The
broader branch is not globally green; the known and newly exposed failure
clusters are documented below.

## Requested work

1. Inspect recent fixes on the all-fixes branch for overlap with new work on
   `upstream/extended`.
2. Remove changes that should no longer be carried independently.
3. Test the cleaned branch before merging, accepting that the intermediate
   branch might fail tests.
4. Merge the latest `upstream/extended`, resolve semantic conflicts, and test
   the result.
5. Run a broad set of even mildly relevant tests, including GPU and corpus
   coverage.

## Cleanup before the merge

Two fixes were removed because `extended` now provides the preferred
implementation:

### Forced in/out connector behavior

All-fixes carried local connector-forcing behavior. The fetched `extended`
branch now implements the intended behavior in its newer connector and
canonicalization flow. Keeping both implementations would have duplicated the
policy and made later behavior depend on pass ordering.

Decision: remove the local implementation and retain its regression coverage.

### Unsigned gather-index handling

All-fixes broadly admitted unsigned gather-index types. `extended` now has the
more precise `GATHER_INDEX_DTYPES` policy covering `int32`, `int64`, `uint32`,
and `uint64` explicitly.

Decision: remove the broad local implementation, use the upstream type set,
and retain the stronger validation tests.

The cleanup was committed as `0745543c6`.

## Pre-merge baseline

The cleaned all-fixes branch was tested before merging so later failures could
be classified where possible.

- Focused overlap tests: 15 passed; 5 expected failures demonstrated the
  missing upstream implementations.
- Python backend, non-GPU: 885 passed, 1 skipped, 9 xfailed, 9 failed, and 58
  deselected.
- The nine Python-backend failures consisted of six stale diagnostic-string
  expectations and three cases containing tasklets unsupported by the Python
  backend.
- A broad non-GPU vectorization run reached 205 passes before it was stopped;
  six CloudSC failures had already appeared.

This baseline later established that the nine Python-backend failures are not
merge regressions.

## Merge and conflict resolution

`upstream/extended` was merged with `--no-ff`. The merge commit is
`f78656774`, with parents `0745543c6` and `f696d113a`.

The main semantic resolutions were:

- `dace/codegen/dispatcher.py`: retained Python code streams while adopting
  the stricter `CodegenError` behavior from `extended`.
- `dace/libraries/tileops/nodes/tile_binop.py`: retained AST utilities and the
  newer C++ expression conversion support.
- `dace/libraries/tileops/nodes/tile_load.py` and `tile_store.py`: combined
  cuTile code-generation helpers with the explicit upstream gather-index type
  policy.
- `dace/transformation/passes/canonicalize/fuse_consecutive_loops.py`: adopted
  the newer source/destination key handling.
- `dace/transformation/passes/vectorization/enums.py`: retained both
  `CUDA_WARP` and `CUTILE` targets.
- `dace/transformation/passes/vectorization/utils/pass_invariants.py`: kept the
  precise lane-dependent transient invariant and added the newer scalar-tasklet
  and nested-SDFG invariants.
- `dace/transformation/passes/vectorization/vectorize_multi_dim.py`: combined
  ternary normalization and TileIota support with CUDA-warp support, upstream
  OpenMP cleanup, and the cuTile assumption-guard gate.
- `dace/transformation/passes/canonicalize/pipeline.py`: combined the newer
  reduction pipeline and progress sweep with the local `assumption_guard`,
  `reduction_to_wcr_map`, and `dump_dir` controls.
- The canonicalization design conflict was resolved in favor of the newer
  upstream design text.
- Gather tests retained the stronger upstream dtype coverage without duplicate
  tests.

## Integration fixes made while resolving the merge

### Self-contained loop-to-reduce seed handling

`extended` removed `accumulator_to_map_and_reduce.py`, but all-fixes'
`loop_to_reduce.py` still imported private helpers from it. The required
functionality was made self-contained in `loop_to_reduce.py`:

- `_RenameWCRConnectors`
- `_wcr_combine_code`
- `_reduction_identity_for`
- explicit Reduce-to-transient and seed-combine construction

The previously failing seed-handling integration then passed.

### Preserve the vectorizer's canonicalization contract

The new vectorizer-entry canonicalization initially enabled
`reduction_to_wcr_map`, changing the graphs expected by the existing
vectorization pipeline. The entry contract now explicitly sets:

```python
ENTRY_CANONICALIZE_KWARGS = {
    'semantic_lifting': False,
    'reduction_to_wcr_map': False,
}
```

This restored the pure LU integration behavior.

### Forward the assumption guard

The new entry canonicalization did not forward the vectorizer's dynamic
`assumption_guard` setting. Forwarding `self._assumption_guard` prevented a
C++ assumption-check tasklet from reaching the Python/cuTile backend and
restored the symbolic LU GPU integration.

### Test compatibility adjustments

- Readable formatting can wrap generated Python expressions, so the cuTile
  code-generation assertion now normalizes all whitespace rather than spaces
  only.
- The ISA-dispatch unit test now mocks host ISA availability. It tests mapping
  independently from the newer architecture-native enforcement.
- Vectorizer stage-order tests now assert both the reduction-to-WCR gate and
  assumption-guard forwarding.

## Validation results

### Focused and tile-operation coverage

- Superseded-fix regression tests: 20 passed.
- Merge-targeted canonicalization, widening, TileIota, expression, and
  orchestrator tests: 54 passed, 2 skipped.
- Final focused merge suite: 40 passed.
- Full non-GPU tile-operation library-node suite: 257 passed, 1 skipped.
- Tile-operation dispatch focus: 24 passed.
- Symbolic LU pure integration: passed.
- Syntax compilation excluding vendored Python-2 code: passed.

### GPU and cuTile coverage

The host GPU was an NVIDIA GeForce RTX 5060 Ti.

- Full `test_cutile_integration.py`: 72 passed.
- Symbolic LU cuTile integration: passed.
- Key GPU/cuTile integrations in total: 73 passed.
- AOT, scalar device arguments, nested SDFGs, storage copies, grid handling,
  offsets, matmul, tasklet rewriting, and instrumentation had broad passing
  coverage in the larger runs.

### Complete Python-backend suite

Result: 942 passed, 2 skipped, 9 xfailed, and 9 failed out of 962 cases.

The failures exactly match the pre-merge baseline:

- Six tests expect old stale-ID wording such as `state with id 6`; current
  diagnostics say `<unresolved state id 6>`.
- Three `test_simple_program` parameterizations contain non-Python tasklets;
  the Python backend correctly reports that it only supports Python tasklets.

Conclusion: no new Python-backend failure was introduced by this merge.

### Loop-to-reduce

Result: 53 passed and 4 failed.

The same four failures were reproduced on a pristine detached worktree at
`upstream/extended`:

- per-row multidimensional reduction using the Reduce library node
- interleaved dual strided reductions
- split two-strided-loop reduction using the Reduce library node
- Reduce lifting inside a nested SDFG

Conclusion: these are inherited `extended` failures, not merge regressions.

### Broad vectorization testing

A combined request collected 5,031 tests spanning vectorization, Python
backend, canonicalization, loop-to-reduce, and tile operations. Vectorization
was exercised in several fresh pytest processes because native aborts prevented
a trustworthy single aggregate.

Positive coverage included large passing blocks for:

- vectorization analyses and lane/tile classification
- cuTile expansions, data copies, nested SDFGs, offsets, rank normalization,
  matmul, and integration tests
- vectorizer orchestrator and walker end-to-end paths
- tasklet conversion, reductions, widening, masks, remainders, and CUDA-tile
  lowering
- many NPBench, PolyBench, TSVC, TSVC 2.5, and CloudSC kernels

The completed final unit tail reported 151 passed and 6 failed. Its failures
were:

- a branch-condition formatting expectation
- two ICON cases where the scatter guard rejects a 2-D index array
- a tile-operation ordering test whose kernel no longer vectorized
- two tile-dependent branch tests whose expected one-dimensional map was no
  longer present

Additional failure clusters appeared in CPU/GPU reduction parameterizations,
staging/scatter paths, symbolic walker cases, and corpus kernels. Because the
processes aborted before pytest session finalization, exact aggregate counts
and all parameter IDs are unavailable and should not be inferred from progress
output alone.

### Partial canonicalization suite

The combined canonicalization trees collected 1,171 cases. Testing was stopped
for time during an expensive CloudSC integration after approximately 39%.

Result at interruption: 445 passed, 19 failed, and 3 xfailed.

Failure concentration:

- 15 wavefront-skew cases, mostly because the transform did not fire or did not
  expose the expected parallel map; one of these additionally lacked an ISL
  implementation
- 2 cases where connector `b` collided with an existing symbol/array name
- 2 guarded-loop split tests expecting a different split-point representation

Most tested pipeline, reduction, transpose, stream-compaction, untile,
stage-order, recurrence, rotation, and real-world pattern cases passed.

### Native aborts

Three large vectorization/corpus processes ended in `SIGABRT`:

1. During `test_nest_reduction_is_idempotent`. The exact test passed when
   rerun alone, indicating process-state-dependent instability.
2. During a compiled PolyBench `CompiledSDFG.fast_call`.
3. During a compiled TSVC `CompiledSDFG.fast_call`.

These aborts prevented normal pytest summaries and are a test-process/native
runtime stability concern. They do not by themselves identify a deterministic
merge regression.

## Formatting and repository hygiene

- Targeted pre-commit hooks on all conflict-resolution and integration files
  pass.
- The all-files hook found broad pre-existing style debt: Ruff proposed 42
  automatic fixes and YAPF rewrote 47 unrelated files, about 2,700 changed
  lines.
- The merge-conflict hook also mistakes reStructuredText `=======` section
  headings in `untile_loops.py` for conflict markers.
- Those unrelated hook rewrites were discarded. No mass formatting change was
  included in the merge.
- The branch had no unresolved paths or unstaged source changes when the merge
  was committed.

## Risk assessment

The central merge resolutions are supported by strong focused and end-to-end
coverage. In particular, cuTile code generation/runtime, the Python backend,
tile operations, vectorizer entry configuration, and assumption handling are
working across CPU and GPU tests.

Remaining risk is concentrated in broader `extended` behavior:

- wavefront skewing and reconstruction
- scatter canonicalization for multidimensional index arrays
- CPU/GPU reduction parameterizations
- a subset of staging and symbolic vectorization patterns
- corpus-scale native execution stability

Only the Python-backend and loop-to-reduce failures were fully baselined and
classified as pre-existing/inherited. Other broad-suite failures require a
dedicated follow-up comparison against pristine `extended` before assigning
ownership.

## Recommended follow-up

1. Run failing broad-suite cases one node ID per process to avoid losing
   diagnostics to `SIGABRT`.
2. Compare vectorization and canonicalization failures against pristine
   `upstream/extended` worktrees before changing merge code.
3. Prioritize the 2-D scatter-guard failures because they arise at the new
   vectorizer-entry canonicalization boundary.
4. Separate optional-ISL failures from wavefront transformation failures.
5. Update the stale Python-backend diagnostic expectations or explicitly
   preserve compatibility wording.
6. Decide whether the three non-Python tasklet cases should be skipped or
   converted to Python tasklets.

## Final repository state

The report and completed merge are published on:

`codex/cutile-all-fixes-extended-merge`

No changes were pushed to the original `agent/cutile-all-fixes` branch during
report publication.
## Current-upstream refresh (2026-08-10)

The integration branch was refreshed from
`origin/codex/cutile-all-fixes-extended-merge` at
`8c06333a985802fe498ea5abda394e7465f77082` to the current
`upstream/extended` tip
`dd5d544e7554ae05fb24569c0e5cf1bae2e34bf7`. The work was performed on
`codex/cutile-all-fixes-extended-current` as a no-fast-forward merge.

### Conflict decisions

Eight textual conflicts were resolved semantically:

- `tile_iota.py`: kept explicit pure/cuTile implementations and disabled
  automatic implementation selection.
- `validation.py`: combined Python-root backend handling with upstream symbol
  assumption collision validation.
- `empty_state_elimination.py`: adopted the upstream free-symbol dependency
  analysis.
- `canonicalize/pipeline.py`: gated the complete reduction normalization stage
  with `reduction_to_wcr_map`.
- `map_predicates.py`: retained both tiled-parameter branch-dependence checks
  and foreign-language tasklet rejection.
- `vectorize_multi_dim.py`: retained the assumption guard, cuTile/CUDA-warp
  targets, all remainder strategies, and the vectorizer entry contract:
  `semantic_lifting=False`, `reduction_to_wcr_map=False`, and
  `unroll_limit=0`.
- The two conflicting regression files were combined to cover both sides of
  their corresponding resolutions.

### Overlap bugs found during semantic review

Three non-textual overlaps required fixes:

1. `TileFMA` was absent from `TILEOPS_NODE_TYPES`. That made the vectorizer
   treat a generated tile node as opaque during later analyses. It is now in
   the central registry and covered by a regression assertion.
2. Upstream residual scalar FMA emission produced an ambiguous
   `fma(half, half, half)` call in CUDA. Tasklet and symbolic C++ emission now
   use `dace::math::fma`; the runtime helper preserves integer semantics,
   widens sub-float operands for a single rounded operation, and forwards normal
   floating-point overloads to `std::fma`.
3. The GPU wrapper documentation and remainder enums specified
   `BRANCHED_MASKED_TAIL` as the K=1 default, but the implementation selected
   `BRANCHED_TAIL`. The wrapper now selects the masked-tile branch and the
   structural expectations account for both generated arms.

Stale-ID validation tests were also updated to the current explicit
`<unresolved ... id N>` diagnostic contract.

### Validation of the refresh

Passing coverage included:

- conflict-focused tests: 153 passed;
- tile-operation library-node tests excluding one reproduced strict-precision
  baseline failure: 454 passed, 4 skipped;
- Python backend: approximately 948 passed, 2 skipped, 9 xfailed, with only
  reproduced baseline failures;
- FMA fusion and runtime coverage: 10 passed, including CUDA fp16;
- branched-tail CUDA coverage: 19 passed;
- GPU reduction and vectorizer coverage: 30 passed;
- the final non-corpus vectorization tail: 319 passed, 55 skipped.

The broad vectorization sweep was run in chunks because native aborts and
optional library failures prevent a reliable single aggregate. Every focused
failure from the final tail was rerun on the pre-refresh integration tip and
reproduced there: four reduction-via-map expectations, one three-way branch
flattening expectation, and two multidimensional scatter-guard cases.

Other inherited or environmental failures reproduced during the sweep include
strict Jacobi precision, Python-backend rejection of a C++ reduction tasklet,
OpenMP reduction code-shape assertions, multidimensional gather/scatter
indexing, tile-store ordering, constant-only staging, TRMM correctness, and
missing CBLAS/MKL. N-body canonicalization still fails on both revisions, but at
different stages (current: invalid View during LoopToReduce; previous: missing
MKL at link time), so that path remains a refresh risk rather than a passing
regression test.

No result above relies on a failure being merely assumed pre-existing: the
listed structural failures were compared directly against a temporary worktree
at the pre-refresh integration commit.

## Newest extended merge (2026-08-28)

The integration branch was refreshed again from the exact checked-out head
`5ad884f78665d7d6daab3b6d4c6ef8fdfe9853b3` to the newest fetched
`upstream/extended` tip
`cf1884c76d87e05c29a2bc6069475975471cdabf`. The merge base was
`dd5d544e7554ae05fb24569c0e5cf1bae2e34bf7`.

The checked-out target was
`codex/cutile-all-fixes-extended-merge`, tracking the same branch on `origin`.
The older remote `origin/agent/cutile-all-fixes` remained at `89a2bf31d` and
was not modified. The source was merged with `--no-ff --no-commit`, inspected,
reconciled, tested, and only then committed. Nothing was pushed.

### Scope inspected before merging

The source range contained 685 commits, including 90 merge commits, and
changed 706 files (about +55,300/-10,516). The target side changed 159 files
from the merge base (about +43,185/-550). Forty-one paths were changed on both
sides. `git cherry` found no byte-for-byte patch-equivalent target commits,
but the semantic audit identified several independently implemented or
superseded fixes before the merge was started.

The virtual merge forecast 14 textual conflicts. The completed integration,
including targeted reconciliation fixes and tests, changes 714 paths relative
to the pre-merge target (+55,678/-10,760) before this report entry.

### What was preferred from newest extended

Newest extended was used as the default for its evolved implementations:

- the reorganized canonicalization/CPU-specialization pipeline, including
  `PinCarriedTopLevelLoops`, early memlet propagation, updated cleanup, scan
  recognition, loop distribution, statement splitting, and stronger
  dependence analysis;
- control-flow-aware accelerator offloading and current GPU placement, stream,
  launch, CUB error, and CUDA/HIP code-generation fixes;
- packaged Copy and Fill library nodes, pure Solve and Cholesky algorithms,
  deterministic ordered connectors, name-aware shape comparison, and newer
  BLAS/linalg stride handling;
- current symbolic replacement, Python floor/modulo semantics, MPR dialect
  printing, readable code generation, explicit-copy lowering, and packaging;
- the evolved `GPUTransformSDFG` handling for interstate-read data rather than
  the target branch's older post-pass scan;
- extended's current SplitTasklets, connector, gather-index, and K=1 GPU
  remainder behavior where those superseded older target-side versions.

### Target-side behavior deliberately preserved

The following target behavior was treated as non-negotiable and retained on
top of extended:

- `BackendLanguage.Python`, `InstrumentationType.PythonTimer`, the cuTile
  schedule/storage enums, Python compilation path, Python target tree, and
  successful-exit synchronization;
- `VectorizeCuTile`, the GPU-first cuTile pipeline, `ISA.CUTILE`, all cuTile
  tile-operation expansions, and the complete tile-node registry including
  `TileFMA` and `TileIota`;
- CuPy BLAS, linalg, and Reduce implementations alongside extended's pure and
  vendor implementations;
- the vectorizer entry contract: `semantic_lifting=False`,
  `reduction_to_wcr_map=False`, `unroll_limit=0`, dynamic assumption forwarding,
  and `assumption_guard=False` for cuTile;
- explicit seeded Reduce lowering with the operation identity followed by a
  combine with the original accumulator seed;
- tile-node symbol replacement hooks, masked transient-update ITE construction,
  conditional reaching definitions, loop-invariant helper-map refusal,
  TileIota materialization, opposite-endpoint widening, direct TileStore
  staging, and cuTile tile-storage recovery;
- Python-backend GPU-global validation, nested descriptor shape/stride
  rewriting, dtype-cast rewriting, direct tasklet edges, view-alias handling,
  cuTile scalar/symbol device-array bridging, nested control-flow scalars, and
  offset/rank handling;
- `RefineNestedAccess` descriptor-shape updates, empty-state reverse-dependency
  protection, and symbol/data collision handling during multistate inlining.

### Semantic overlap decisions

Several bug fixes existed in both lines in different forms:

- **Interstate GPU data:** extended's `_move_to_gpu`-based implementation
  superseded the target's later post-Step-6 scan. The extended design was kept;
  the target regression intent was retained and updated to the new host-pinning
  contract.
- **FMA ambiguity:** extended already routed tasklet FMA through
  `dace::math::fma` and added low-precision overloads. That implementation was
  kept, with the target's symbolic-runtime qualification and exact integral
  `a*b+c` overload added where extended's generic floating implementation could
  lose integer precision.
- **K=1 GPU remainder:** both sides independently selected the branched masked
  tail. Extended's current structure and test wording were preferred.
- **Connector/gather policy and SplitTasklets fixes:** the extended versions
  were newer and were kept. Older target connector-forcing work that had
  already been deliberately removed was not revived.

### Textual conflict resolutions

All 14 conflicted paths were resolved manually:

1. `dace/codegen/codegen.py`: combined extended's explicit-copy, SIMD-marking,
   compiler-warning, translation-unit, and CUDA preprocessing with the target's
   Python backend dispatch and `CodeObject` creation. C++-only passes are
   explicitly guarded so they cannot mutate Python/cuTile SDFGs.
2. `dace/codegen/cppunparse.py`: used extended's current real/imaginary/FMA and
   dialect logic, removed a duplicate `fma` table entry, and retained the
   runtime helper route.
3. `dace/libraries/blas/nodes/axpy.py`: kept extended's ordered connectors and
   value-aware symbolic checks together with the target's CuPy expansion.
4. `dace/libraries/blas/nodes/ger.py`: made the same union for GER, retaining
   CuPy while taking extended's deterministic interfaces and comparisons.
5. `dace/libraries/linalg/nodes/solve.py`: kept extended's real pure Gaussian
   elimination, rank-one RHS, restriding, and validation logic, plus the
   target's CuPy/vendor implementations. The obsolete target pure stub was
   removed rather than leaving duplicate classes.
6. `dace/runtime/include/dace/math.h`: retained extended's low-precision FMA
   placement and generic integral `pow`; added a same-type integral FMA
   overload so integer tasklets retain exact integer semantics.
7. `dace/sdfg/replace.py`: retained extended's SDFG-aware property replacement
   and `SymExpr` handling, then invokes the target's `node.replace_dict` hook
   for tile-node private expressions.
8. `dace/sdfg/validation.py`: followed extended in removing the retired symbol
   assumption-collision framework, while preserving root-backend detection,
   Python-host access to `GPU_Global` CuPy arrays, caching, and correct
   unresolved source locations.
9. `dace/symbolic.py`: kept extended's dialect-aware casts and standalone MPR
   spelling; runtime FMA prints as `dace::math::fma`.
10. `dace/transformation/passes/canonicalize/pipeline.py`: used extended's new
    stage bodies and ordering, including `PinCarriedTopLevelLoops` and
    `PropagateMemlets`, but retained the target's complete
    `reduction_to_wcr_map` gate and optional assumption guard.
11. `dace/transformation/passes/vectorization/vectorize_multi_dim.py`: used
    extended's current organization and diagnostics while preserving CUTILE,
    TileFMA/TileIota awareness, lazy imports, assumption forwarding, and the
    guarded entry canonicalization configuration.
12. `tests/passes/vectorization/gpu_reduction_block_atomic_cudatest.py`: kept
    extended's evolved GPU reduction expectations while retaining coverage of
    the target's supported reduction behavior.
13. `tests/passes/vectorization/test_vectorize_gpu.py`: kept extended's detailed
    two-arm documentation and expectations while preserving the masked-tail
    default.
14. `tests/transformations/interstate/test_inline_multistate_sdfg.py`: retained
    both independent regression families: target symbol-versus-array/constant
    collisions and extended library/tasklet connector-interface behavior.

### Silent clean-merge audit

The high-risk shared paths were reviewed even when Git reported no conflict.
This found and removed obsolete duplicate pure stubs in `cholesky.py` and
`inv.py`. The final implementation registries contain both pure/vendor and
CuPy choices as intended. The audit also confirmed:

- Copy/Fill imports were migrated to the new package layout;
- Python/cuTile enums, compiler configuration, and backend dispatch survived;
- every tile operation is in `TILEOPS_NODE_TYPES`;
- Reduce retains its CuPy implementation;
- explicit reduction identity/seed-combine lowering survived;
- `RefineNestedAccess`, tile expression replacement, masked updates, TileIota,
  widening, storage recovery, and nested-symbol collision fixes survived;
- no new duplicate class definitions remained (apart from the pre-existing
  ONNX `PureExpand` pattern).

### Repairs made after integration testing

Testing exposed six integration issues. They were repaired without backing out
extended's architecture:

1. **Host-pinned GPUTransform producers.** Extended correctly keeps
   Default-stored interstate-read data on the host, but two free-tasklet phases
   still offloaded its producer. Both phases now honor `host_data`. Tests cover
   the default-host case structurally and an explicit `GPU_Global` case through
   GPU compile and execution.
2. **Python Reduce auto-selection.** Extended schedule inference could choose
   an OpenMP expansion inside a Python-backend SDFG, leaving a C++ tasklet in
   generated Python. `ExpandReduceAuto` now checks the root backend and selects
   the pure expansion for Python.
3. **Semantic cuTile store discovery.** Incoming offloading/name allocation
   changed names such as `C_tile_out` to `C_gpu_tile_out`. The same-storage
   runtime tests now inspect the `TileStore._src` edge instead of a suffix.
4. **Supported cuTile pipeline in copy tests.** Six structure tests manually
   reconstructed the obsolete vectorize-before-GPU order; extended then
   restamped the tile mask `GPU_Global`. They now use the single
   `VectorizeCuTile` front door with canonicalization skipped only because the
   fixtures are already canonical.
5. **Name-independent staging detection.** Extended's default offloader names
   device staging arrays `x_gpu`; legacy `GPUTransformSDFG` can still name them
   `gpu_x`. `VectorizeCuTile(use_gpu_storage=True)` no longer infers staging
   from either spelling. It detects pre-existing host/device staging edges
   before storage promotion, separately reports unpromoted descriptors, and
   avoids mistaking normal GPU-to-GPU result copies for staging. Runtime tests
   accept the cuda.tile version-dependent `ValueError`/`RuntimeError` taxonomy
   while preserving the device-array contract.
6. **Float64 baked-in constants.** The full GPU suite exposed the known
   cuda.tile demotion of `0.33333` to float32 in Jacobi. The narrowly scoped
   `_cutile_f64_const_tile` implementation and its fail-closed tests were
   restored from local cuTile development commit `136880349` (the commit was
   not cherry-picked). Non-float32-exact float64 literals are reconstructed
   from two exactly representable terms; comparisons infer the operand dtype,
   not a boolean output dtype. Both symbolic-size Jacobi integrations now pass
   at the original float64 tolerance.

Additional regression coverage was added for integer FMA above the exact
double range and MPR/runtime FMA dialect spelling.

The workspace `AGENTS.md` was also updated with the durable rule that GPU
staging must be detected semantically: current and legacy offloaders use
different name conventions.

### Dependency setup used for validation

The extended test tree required `ordered-set`, `z3-solver`, and `islpy`; these
were installed into the existing shared `/venv/main` environment. No new
virtual environment was created and no source dependency manifest was changed
for this purpose.

### Validation results

Passing coverage after reconciliation included:

- pre-merge focused preservation suite: 136 passed (the original 134 plus two
  new regressions);
- full tile-operation/cuTile suite: 463 passed, 6 expected skips;
- full Python backend: 951 passed, 2 expected skips, 9 expected failures;
- focused canonicalization pipeline/stage/knob/end-to-end gate: 94 passed;
- final post-format reconciliation gate across Python, C++, pure tileops,
  cuTile GPU, BLAS, vectorizer, and transformation paths: 276 passed, 1
  expected skip;
- `test_vectorize_gpu.py`: 15 passed;
- focused MPR/FMA/canonicalization coverage: 87 passed;
- runnable pure/CuPy/cuSolver linalg cases: 13 passed; 30 known Inv cases
  skipped;
- targeted pre-commit hooks on all 24 manually reconciled conflict and repair
  files: passed (`ruff`, conflict marker, EOF, trailing whitespace, and YAPF).

The GPU block-reduction file passed 13 cases in one process before a cumulative
CUDA process segfault occurred at the `max` case. The remaining `max` and `min`
cases each passed in fresh processes, so every parameterization was exercised
successfully; the aggregate-process instability remains.

Twelve optional MKL/OpenBLAS linalg cases failed only because this environment
lacks `mkl.h`/`cblas.h`. They were not treated as merge regressions.

### Broad vectorization baseline classification

A monolithic non-lib-node vectorization run was stopped at 12% to preserve
diagnostics before the known native-process instability could discard them.
At interruption it had 294 passes and eight failures, all in the CPU OpenMP
reduction code-shape assertions.

The exact current file result is 4 runtime passes and 8 structural failures.
A detached worktree at the exact pre-merge target reproduced the same eight
structural failures. Its four runtime cases additionally failed to compile
only because the temporary worktree did not have the `moodycamel` submodule
checked out. Therefore the eight reported failures predate this refresh; the
merged main worktree's numerical reduction integrations pass.

### Remaining risk

The merge-critical Python/cuTile, tile-operation, canonicalization entry, GPU
storage, FMA, validation, and library paths have broad structural and runtime
coverage. Remaining risk is concentrated in known pre-existing or
environmental areas:

- CPU OpenMP reduction shape expectations;
- optional MKL/OpenBLAS builds on this host;
- cumulative long-process CUDA stability;
- the wider corpus/wavefront/scatter issues already documented in the earlier
  sections of this report.

No known failure was classified as inherited without either a direct detached
baseline comparison or a concrete missing-runtime/header diagnosis.

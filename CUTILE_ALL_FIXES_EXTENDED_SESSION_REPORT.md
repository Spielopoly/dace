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

# Canonicalize soundness bug: order-dependent miscompile of in-place stencils

**Status: OPEN — hand-off documentation.** Found 2026-07 while investigating the
flaky `tests/passes/vectorization/lib_nodes/test_cutile_store_offset.py::test_jacobi_1d_end_to_end`.
The affected test now runs with `canon=False`; a skipped `_canonicalized` variant
preserves the repro. Do not re-enable it until this bug is fixed.

## Symptom

For the in-place two-statement stencil

```python
@dace.program
def _jacobi_1d(TSTEPS: dace.int64, A: dace.float64[N], B: dace.float64[N]):
    for _ in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])
```

`canonicalize(prog.to_sdfg(simplify=True))` produces, on *some process
trajectories*, an SDFG that computes the wrong `B` (every interior element off
by ~1e-2 for float64 inputs in [0,1); `A` stays exactly correct). On other
trajectories of the *same input* it produces a correct SDFG. Observed failure
rate of the pytest file: roughly 2 in 3 runs.

## Evidence chain (all verified experimentally)

1. The frontend/simplify input SDFG is deterministic (identical fingerprints in
   clean and failing runs): one dataflow state per loop iteration containing both
   statements with correct dependency edges. Compiled directly (C++ CPU backend)
   it matches NumPy.
2. On a failing trajectory, the **post-canonicalize** SDFG already computes the
   wrong result when compiled with the plain **C++ CPU backend** — no
   vectorizer, no tileops, no Python backend involved.
3. The same post-canonicalize SDFG pushed through `VectorizeCuTile` + the
   Python/cuTile backend produces **bit-identical wrong values** to the CPU run
   (`cpu == gpu` exactly). Everything downstream of canonicalize is faithful;
   the miscompile is canonicalize's alone.

## The wrong canonical form

Correct trajectories fuse the loop body into one state where the (duplicated)
`B`-write is ordered *before* the `A`-update, so the rewrite stores the same
value (benign). Failing trajectories end with (states in execution order):

```
LOOP for (_loop_it_0 < TSTEPS):
  STATE 1: fused maps: B[1:N-1] = 0.33333*(A[i]+A[i+1]+A[i+2]);
           A[1:N-1] = 0.33333*(B[i]+B[i+1]+B[i+2]); transients updated   # correct
  STATE 2: A_slice_plus_A_slice[i]         = A[i] + A[i+1]      # A is ALREADY UPDATED
           A_slice_A_slice_plus_A_slice[i] = ... (+ A[i+2])     # stale/new mix
           B[i+1] = 0.33333 * A_slice_A_slice_plus_A_slice[i]   # REWRITES B — WRONG
```

The pipeline manufactures a *duplicate* of the `B`-statement (via the
fission/rotation of the fused body into per-op loops) and on failing
trajectories schedules it *after* the `A`-update, where its operands are a mix
of pre- and post-update `A`. The duplicate is benign only when scheduled before
the `A`-update — which is exactly what the "lucky" trajectories do. Mid-pipeline
observations (single trajectory, `_build_stages(peel_limit=4,
break_anti_dependence=True, interchange_carry_with_map=True)`): the ordering
divergence is first visible at the `lower` stage
(`PatternMatchAndApplyRepeated([MapToForLoop])` — maps are lowered to loops in a
different order), and the stale-operand rewrite materializes around the
`loop_to_x` / `fuse` stages.

## Why lowering is not a pure function of the input (nondeterminism source)

The pipeline outcome depends on process-global iteration order, in two ways,
both verified:

- **`PYTHONHASHSEED`**: with prior in-process lowerings (the other tests in the
  file) and everything else fixed, seeds {0, 2} reproduced the miscompile and
  {1, 3, 4} did not — each seed deterministically. Hash-randomized `str`/sympy
  `Symbol` hashing changes set/dict iteration order consumed (transitively) by
  pattern-match enumeration (`match_patterns` yields VF2 matches in container
  order) and by name generation.
- **Allocation history**: an unrelated ~15-line edit to the *driver script*
  (same seed, same pipeline input) flipped fail→pass, and running the same
  stage prefix in processes that differed only in later argv produced different
  intermediate forms. Some ordering consumed by the pipeline is id()/address
  based (sets of graph objects), i.e. a function of every allocation made since
  process start. This is why the test is order-dependent within the pytest
  file: the preceding tests' lowerings change the allocator/hash state, not any
  SDFG content.

Repro (fresh process per seed; artifacts land in `outdir`):

```python
# 1..4: lower the other test programs first (poisons allocator/hash state)
for _ in range(4): lower_cutile(_store_offset_1d, (8,))
for _ in range(4): lower_cutile(_load_offset_1d, (8,))
# 5: jacobi through the real canonicalize stage list, then CPU-compile
sdfg = _jacobi_1d.to_sdfg(simplify=True)
canonicalize(sdfg)                    # CPU presets
csdfg = sdfg.compile()                # default C++ backend, NO vectorization
# compare against NumPy for n in (32, 34): B interior wrong on bad seeds
```

Sweep `PYTHONHASHSEED=0..4`; at the time of writing seeds 0 and 2 failed with
`maxerr(B) = 1.964e-2` (n=32) / `8.877e-3` (n=34). Because the trigger also
depends on allocation history, the exact failing seeds may shift with any code
change; the pytest file itself (`test_cutile_store_offset.py` with the jacobi
test set back to `canon=True`) failed ~2/3 of unseeded runs.

## Suggested fix directions (for the canonicalize owners)

1. Soundness: whatever duplicates the `B`-statement and later reorders it across
   the `A`-update violates the anti/flow dependence between the two statements
   (`B`-write reads `A`; `A`-write follows). The reordering/fusion legality check
   that admits the late duplicate is the actual bug.
2. Determinism: pattern-match application order should be derived from a
   canonical ordering (e.g. node/state creation ids), never from `set`/`dict`
   iteration over hashed strings, sympy symbols, or object ids. That would make
   failures reproducible even if soundness bugs remain.

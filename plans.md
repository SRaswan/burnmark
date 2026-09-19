# Plans & Roadmap

Design document for where BurnMark is going. Read alongside [`bugs.md`](bugs.md) for current bug status and the architecture section of [`README.md`](README.md) for codebase layout.

---

## Phase 0 — deepen the generator

Four places where the fuzzer's reach is currently bounded:

**Type-directed generation.** `AutogradProgram` has a shape-aware builder that only emits legal ops. `TensorProgram` doesn't — it derives `Arbitrary` and fixes up illegal operands after the fact (`resolve_broadcast_compatible` etc.), wasting generation budget and distorting op distribution. Extend the builder to both program types so nothing is generated and discarded.

**Higher-rank tensors.** `Shape2` is hardcoded 2-D; the interpreter is `Tensor<2>` throughout. Burn's `Tensor<B, D>` is const-generic, so supporting rank 1–5 means a rank-erased dispatch enum over `D` plus a dynamic `ShapeN`. This matters: `swap_dims`, `flip`, `narrow`, per-dim reductions, and broadcast edge cases all live in higher rank, and the original 0.20.1 bug was a `swap_dims` bug.

**Op coverage.** Current set in `src/ir/ops.rs`. Highest-value additions: `swap_dims`/`permute`, `reshape`, `slice`/`slice_assign`, `gather`/`select`, `mask_fill`/`mask_where`, `softmax`/`log_softmax`, `min`/`max`/`clamp`, tensor–tensor `powf`, `recip`, `var`/`std`, `prod`, `cumsum`, `sort`/`argmax`, `stack`/`chunk`, `tril`/`triu`, `erf`, `sin`/`cos`.

**Special-value seeding.** `bytes_to_floats` maps seeds uniformly into `[-1, 1]` — it never directly produces `NaN`, `±inf`, subnormals, signed zero, or exact `±1` boundaries. Three of the five bugs found so far are special-value dependent. A seeded special-value pool mixed into leaf data is probably the cheapest yield increase in the whole roadmap.

Supporting changes:

- **Campaign discipline.** `-max_total_time` / `-jobs N` for parallel runs, `cargo fuzz cmin` to keep `fuzz/corpus/` from bloating, `cargo fuzz coverage` to drive op-coverage work from actual gaps rather than guesses.
- **Oracle tolerance tracking.** The `recip` bug was a ~0.2% relative error that `macerator`'s test suite missed at `2^-8` tolerance. Track tolerance deliberately; maintain an explicit allowlist for ops where last-bit transcendental divergence is genuinely expected.
- **Non-device axes.** The cheapest coverage left may not be a new device at all:
  - `simd` on/off — `burn-ndarray` and `burn-flex` both default to `simd`; a scalar build is a build-axis comparison, relevant to bug #4.
  - `fusion` on/off — fused vs unfused on the same device.
  - `autotune` on/off — autotune picking a wrong kernel.
  - `blas-netlib` — BLAS vs pure-Rust matmul.
  - dtype axis — f64 as a higher-precision reference for f32.

### Adding a backend (measured costs)

Burn 0.22 dropped the `Backend` type parameter — backend is a property of `Device` — so the interpreter is already backend-generic. One `cargo feature` + one `Device::X()` call is enough for a CPU backend. GPU backends need three more things first:

1. **Cross-thread panic hook.** `catch_as_result` uses `catch_unwind`, which only catches panics on the calling thread. A GPU backend that fails a shader exits 0 with a wrong answer — strictly worse than crashing. Needs a global panic hook recording cross-thread panics into a flag the harness checks before comparing.
2. **`OnceLock` for device construction.** ~0.31 s per process on the Metal probe. Device construction must be hoisted out of the per-iteration path.
3. **Per-op tolerance.** `1e-4` is probably too tight for GPU transcendentals against CPU LibTorch.

Measured backend comparison (30 s, `-max_len=8`):

| | flex | cpu (CubeCL) | metal / wgpu | libtorch_mps |
|---|---|---|---|---|
| Cold build | 14 s | 1 m 08 s | minutes | none |
| Runs under ASAN | yes | no (linkme/ASAN — fixable with `-asan-globals=0`) | untested | yes |
| exec/s | 5,334 | 323 (degrading) | untested | untested |
| RSS | flat | climbs (260→562 Mb) | untested | untested |
| Needs OnceLock / panic hook / per-op tolerance | no | no | all three | no |

CubeCL CPU's ASAN issue: `pliron` registers dictionary keys with `linkme`'s `#[distributed_slice]`. Under ASAN, global redzones widen the gap between dupcheck sentinels until the check fires spuriously. Fix: `RUSTFLAGS="-Cllvm-args=-asan-globals=0"` — keeps ASAN, drops only global redzone detection. Measured: 227 exec/s (ASAN) vs 323 (no sanitizer). Any run including `cpu` also needs `-- -rss_limit_mb=8192`.

CubeCL CPU's coverage issue: 30 s produced 4,726 points on ndarray vs **39,898** on cpu — because the instrumented JIT is executing. LibFuzzer will optimise toward inputs stressing CubeCL's compiler, not burn's math. Treat cpu coverage numbers as incomparable.

`sign(NaN)` correctness probe across backends on unpatched `0.22.0-pre.3` (gradient of `abs(log(x))` over `[-0.5, 0.25, 2.0, -3.0]`):

| Backend | Gradient | Correct? |
|---|---|---|
| `libtorch`, `libtorch_mps`, `metal`, `cpu` | `[-0.0, -4.0, 0.5, -0.0]` | yes |
| `ndarray` | `[-2.0, -4.0, 0.5, -0.33...]` | no — `1/x`, sign from NaN's sign bit |
| `flex` | `[NaN, -4.0, 0.5, NaN]` | no — returns the NaN |

CubeCL CPU is the reference: not deprecated, correct on `sign(NaN)`.

---

## Phase 1 — automatic crash characterization

A raw crash artifact is not a filable bug. Moves that are currently manual and should be harness:

- **IR-level minimization.** Delta-debug at the SSA level (drop instructions, shrink leaf shapes, collapse aliasing) rather than bytes — the IR is what the writeup needs to quote.
- **Cluster by root cause.** One bug produced 261 crashes in 5 minutes. Dedup on a signature (panic message + minimized op sequence + divergence shape) to turn a pile of artifacts into a triage queue.
- **Automatic parameter sweep.** The `relu`-chain bug's characterization (needs `NaN`, needs two chained `relu`s, needs `n mod 4 != 0`) came from manually sweeping `n` 16–257. Given a minimized repro, automatically sweep element count, rank, op-repetition count, NaN-vs-finite — and report which conditions are load-bearing.
- **Regression bisect.** `git log -S` on the implicated function + `git bisect` against the repro is scriptable. Naming the causing commit is most of what makes a report credible to a maintainer.

---

## Phase 2 — the fix-and-continue loop

One unfixed bug saturates the crash channel; fix latency *is* discovery rate. Every bug found in 0.22 was found only after hand-patching the previous one. The loop automates the moves that are currently manual:

| Loop step | Current manual equivalent |
|---|---|
| Point fuzzer at patched code | `[patch.crates-io]` in `fuzz/Cargo.toml` |
| Isolate each fix | per-bug `git worktree` |
| Run with several fixes at once | cherry-picked integration branch |
| Pin a dependency whose API drifted | worktree off a tag with fix cherry-picked |
| Keep repros honest | root `Cargo.toml` deliberately not patched |

**Loop steps:**

1. **Run** until crash/divergence; capture artifact.
2. **Characterize** (Phase 1: minimize → dedup → sweep → bisect). If signature matches an already-patched bug, discard and continue before spending anything on triage.
3. **Root-cause** to a specific line. This step must not be skipped — bug #4 is still unfiled precisely because it isn't pinned.
4. **Patch** in a fresh worktree on a fresh branch — one bug per branch, based on upstream `main`.
5. **Validate:**
   - target's own test suite — zero new failures, zero new ignores;
   - new regression test that fails before and passes after;
   - standalone repro re-run against patched tree;
   - original crash artifact replayed against rebuilt fuzz target.
6. **Falsify.** Revert the fix in a scratch worktree and confirm the original artifact crashes again. If it doesn't, the fix was never load-bearing and the real bug is still out there. The loop's worst failure mode is a wrong fix that silences the signal.
7. **Re-point and continue.** Regenerate integration branch, rewrite `[patch.crates-io]`, rebuild, go to 1.
8. **Draft submission.** Writeup into `docs/`, repro into `examples/`, exact `gh` command staged. Loop stops here — filing is the human's call.

---

## Phase 3 — branch topology on a fork

All agent branches live on a fork, never on upstream. Promoting a fix to an upstream PR is a separate human decision:

```bash
gh pr create --repo tracel-ai/burn --head <you>:fix-sign-nan-ndarray --base main
```

**The topology problem:** the fuzzer needs every fix simultaneously; a reviewer needs each fix alone based on upstream `main`. A linear stack can't satisfy both — `fix-powi-scalar-zero-grad` sitting on top of `fix-sign-nan` forced unpicking when promoting powi alone, under exactly the time pressure this project exists to avoid.

**Solution: independent siblings + throwaway integration branch.**

```
fix-sign-nan-ndarray      ─┐   each: one bug, one commit, based on origin/main
fix-sign-nan-flex         ─┤   independently promotable as a PR
fix-powi-scalar-zero-grad ─┤
                           └─→ fuzz-integration   ← what fuzz/Cargo.toml points at
```

```bash
git checkout -B fuzz-integration origin/main
git cherry-pick <tip of each fix branch>    # regenerate from scratch; never commit onto it
```

Every branch is a single commit off `main`, so keeping up with upstream is `git rebase main` per branch with no restacking. Fixes touch different crates and files, so cherry-pick conflicts are unlikely.

Additional benefits:
- **Fork CI is free cross-architecture validation.** Every bug so far is aarch64-SIMD-flavoured; pushing to the fork runs the target's own workflows on hardware we don't own.
- **Cross-repo linking.** The `recip` fix belongs in `macerator` but the symptom is in `burn-ndarray`. A dependency-side PR should ship with a linked issue on the consumer.

Maintain a **campaign ledger**: one row per distinct bug — signature, repro, root-cause line, branch, validation status, filed/unfiled — so "unfiled" is a visible number rather than something that quietly sits.

---

## Phase 4 — the orchestration stack

**Governing principle: deterministic harness, agent only at judgment steps.** Running the fuzzer, watching for crashes, hashing signatures, regenerating branches, rewriting `[patch.crates-io]`, rebuilding — all ordinary code. An agent is invoked for root-cause, patch, regression test, and writeup.

**[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk) as the spine** (Python/TypeScript sidecar shelling out to `cargo fuzz`):

- **Dedup before invoking.** One bug produced 261 crashes in five minutes; only a first-of-signature crash should reach a model.
- **Fresh session per crash** with a pre-assembled context bundle — minimized IR, sweep table, bisect result, implicated source file. Use `resume` only for continuing one specific triage.
- **Hard spend caps:** `max_budget_usd` / `CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS` / `CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH`.
- **File checkpointing** for clean fix rejection.

**Skills** are the highest value per hour here, since the procedures are checklist-shaped and already written in prose: `triage-crash`, `validation-gate`, `write-bug-report`, `patch-and-repoint`. Skills in `.claude/` are loaded by both the unattended loop and interactive sessions — fix one, both improve.

**Hooks** turn non-goals into enforcement: `PreToolUse` refusing `git push --force` and writes to the shared checkout; `PostToolUse` for `cargo fmt` and clippy.

**Subagents** for parallel triage of distinct signatures in separate worktrees, and a **falsifier** whose only task is to break a patch another agent wrote — context isolation means the falsifier doesn't inherit the reasoning that produced the patch.

**MCP:** an in-process SDK server wrapping this project's own harness operations as typed tools (`minimize_artifact`, `sweep_param`, `replay_artifact`, `run_validation_gate`) beats having a model reconstruct long `cargo +nightly fuzz` incantations with the right env vars every time.

**Cheapest starting point:** prototype the loop interactively with a couple skills driven on an interval before writing any orchestrator code.

---

## Prior art

- **OSS-Fuzz / ClusterFuzz:** continuous fuzzing, crash bucketing, bisection, filing, verify-and-close. No fix generation.
- **DARPA AIxCC (Aug 2025):** scored on finding *and* automatically patching vulnerabilities with validation. Closest prior art.
- **Google's LLM-based OSS-Fuzz patching:** generates candidate fixes with human review before submission.
- **DL framework differential testing literature:** CRADLE, LEMON, Muffin, EAGLE, NNSmith, Tzer — typed graph generation and gradient checking.

Three things about this setup look unusual:

- **The oracle is a second implementation, not a sanitizer.** Four of the five bugs here produce no crash — 0.2% wrong numbers, a wrong `-0.0`, a silently-`None` gradient, a gradient on the wrong elements. There is nothing to bucket on.
- **Patch-to-unmask instead of suppress.** You cannot suppress `recip()` being 0.2% wrong without also suppressing `log`, `sigmoid`, and everything composed from them. Fixing is the only way to keep fuzzing.
- **Cross-repo patching across a dependency boundary** — fix in `macerator`, symptom in `burn-ndarray`, injected via `[patch.crates-io]`. AIxCC-style tasks are single-repo.

Defensible claim: *an agentic differential-fuzzing loop where each validated fix is injected back into the dependency graph to unmask the next bug, applied to numerical and autodiff correctness rather than memory safety.*

---

## The human gate and non-goals

The loop stops one step short of submitting. Branch pushed, tests green, writeup drafted, `gh` command printed — then it waits. Filing is a person's call, every time.

Explicit non-goals (belong in hooks, not just this list):
- No auto-merging or auto-filing PRs against upstream.
- No force-pushing to shared branches.
- No editing another session's branch.
- No patching the root `Cargo.toml` (would "fix" `examples/` repros and destroy their purpose).

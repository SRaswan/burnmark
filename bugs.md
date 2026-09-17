# Bugs

Active and past bugs found by this fuzzer, their root causes, fix locations,
and filing status. This is the working log — read it before touching any of
the four bugs below or resuming bug #4's investigation. See `CLAUDE.local.md`
for the project's goal and how to extend the fuzzer itself.

## Current state

Fuzzer targets Burn `0.22.0-pre.3` (newest published; no stable 0.22.0 on
crates.io, `main` is also `0.22.0-pre.3`) and runs clean. The original 0.20.1
`swap_dims` bug is confirmed fixed on 0.22.

Four distinct bugs found since. **All four fixes are written and tested; none is
filed anywhere.** Filing is the bottleneck, and the whole point of this project.

| # | Bug | Owner repo | Fix location | Filed |
|---|---|---|---|---|
| 1 | aarch64 SIMD `recip()` silently loses ~0.2% precision on ≥32 elements (`vrecpeq_f32` with no Newton–Raphson refinement); corrupts `log()`'s gradient and `sigmoid()`'s forward pass | **macerator** (a burn *dependency*) | `~/Documents/Workspace/macerator`, branch `newton`, commits `3de8779` + `f94b31d` | no |
| 2 | `sign(NaN)` returns ±1 from the raw sign *bit*, so `abs(log(x))`'s gradient is `1/x` garbage for `x < 0` | **burn-ndarray** | `~/Documents/Workspace/burn`, branch `fix-sign-nan`, commit `189903ad0` | no |
| 2b | same bug class, independent implementation (`copysign`-based `float_sign` returning the NaN itself) | **burn-flex** | same branch, commit `c11a8a037` — fixed by peer session `burn-f3`, not mine to redo | no |
| 3 | `powf_scalar(0.0)` returns a tensor detached from the autodiff graph (`float_powi_scalar`'s `0 => float_ones(...)` arm is a *constructor*, unrelated to `lhs`); every backend inherits it | **burn-backend** (not deprecated — the strongest of the four) | worktree `~/Documents/Workspace/burn-powi-fix`, branch `fix-powi-scalar-zero-grad` (stacked on `fix-sign-nan`), commit `e090bdd0d` | no |
| 4 | `x.log().relu().relu()` with `x < 0`: gradient wrong in exactly the trailing `n mod 4` elements | burn-ndarray (suspected) | **not root-caused — no fix** | no |

Per-bug detail, root-cause traces, and repros live in `docs/` —
`simd-recip-precision-bug.md`, `powi-scalar-zero-grad-bug.md`,
`relu-chain-nan-simd-remainder-bug.md` — with standalone reproductions in
`examples/`. Don't re-derive any of it from scratch; read the writeup first.

### Filing order (burn-ndarray is deprecated)

`Device::ndarray()` now carries `#[deprecated(since = "0.22.0", ...)]`: burn-ndarray
is slated for removal in favour of burn-flex. Three of the four bugs are in a
backend that is going away, which reorders what to file:

1. **#3 `powf_scalar(0)`** — in `burn-backend`'s default impl, not deprecated,
   inherited identically by every backend. File first.
2. **#2b burn-flex `sign(NaN)`** — in the backend that *replaces* NdArray, so it
   stays relevant. Confirmed present in the published crate, not just `main`.
3. **#2 burn-ndarray `sign(NaN)`** — correct, but a fix to code on its way out.
   Best filed as the second half of #2b (one bug class, two backends) rather
   than alone.

#1 is a `macerator` PR, and should ship with a linked burn-ndarray issue: the
people who *hit* it are burn users who have no idea macerator exists.

### Still open: bug #4, resume here

`x.log().relu().relu()` where `x < 0`, so `log(x)` is NaN throughout. NdArray's
gradient is correct everywhere *except the last `n mod 4` elements*, where it
gives `-0.0` instead of the correct passthrough. LibTorch is correct everywhere.
No panic — silently wrong, size-dependent.

Each condition confirmed necessary by direct experiment: needs NaN (a plain
negative input diverges nowhere); needs **two** chained `relu`s (one alone is
clean at every size tested, including the sizes that fail with two); needs
`n mod 4 != 0` (swept n = 16..257 — zero divergence at every multiple of 4,
exactly `n mod 4` trailing divergent elements otherwise; 4 is the `f32` NEON
lane width here).

Traced as far as: `relu`'s backward masks through `float_lower_equal_elem`,
which *is* macerator-SIMD-accelerated with its own scalar-tail fallback
(`burn-ndarray/src/ops/simd/cmp.rs`) — plausible on its face, except that
fallback runs identically regardless of how many `relu`s precede it, yet the bug
needs two. Leading unconfirmed hypothesis: `relu`'s `.memory_bound()` /
`RetroForward` checkpointing means a second stacked `relu` forces the first's
output to be *recomputed*, and that recomputation's SIMD `clamp_min` may handle
the NaN tail differently. **This is where the investigation stopped** — deep in
burn's checkpointing internals. No fix attempted, because nothing is pinned to a
line yet. Worth filing as an issue with the repro as-is regardless; the
signature is precise enough that someone with checkpointing context could likely
spot it fast.

### Operational gotchas that keep biting

- **`fuzz/Cargo.toml` `[patch.crates-io]`** points at the locally-fixed
  checkouts, so re-fuzzing finds *new* bugs instead of drowning in known ones.
  This is the manual version of the agentic loop's re-point step.
- **Root `Cargo.toml` is deliberately NOT patched.** Every `examples/` repro is
  meant to demonstrate its bug against real published crates — patching root
  would silently "fix" them and destroy their purpose.
- **macerator API drift**: burn-ndarray 0.22 wants macerator `^0.3.4`, but
  macerator `main` renamed `Arch::new()` → `Arch::detect()`. Validating against
  burn-ndarray means a worktree off the `macerator-v0.3.4` tag with the fix
  cherry-picked — not `main`.
- **The shared `~/Documents/Workspace/burn` checkout is not exclusively mine** —
  peer session `burn-f3` works in it. Hence bug #3 living in a separate worktree.
  Never yank that checkout's branch out from under it.
- **`unsafe fn` on macerator's aarch64 impls doesn't compile** (`E0053`): the
  `Simd` trait declares them as safe `fn` via `declare_unop!`, so implementors
  can't unilaterally mark them unsafe. An explicit `unsafe { }` block is the
  correct match for that codebase's style.
- **One loud bug hides all the others.** The recip bug produced 261 crashes in a
  5-minute continuous run, all one root cause; `powf_scalar(0)` produced 640–647
  divergent runs per 25-minute session. Every distinct bug here was found only
  after patching the previous one in.

### Fuzzer changes made along the way

- Op coverage: `Div` and `PowfScalar` added to the IR (`src/ir/ops.rs`,
  `generate.rs`, `interpreter/mod.rs` + `shape.rs`) — which is what turned up #3.
- Size cap: `Shape2::exceeds_cap` / `MAX_TENSOR_ELEMENTS` in `src/ir/shape.rs`,
  wired into both interpreter paths. The plain `TensorProgram` path has no
  shape-aware generator, so chained `Repeat`/`Concat`/`Matmul` could compound
  into a `malloc(4294967296)` OOM abort mid-run. A generation-space gap, not a
  Burn bug.

### Oracle fix worth knowing about

The old `compare_outputs` decided divergence purely by
`abs_diff > 1e-4 * scale`, which is `false` whenever either side is `NaN` or
infinite — so **every** special-value divergence was silently reported as
agreement. Measured on the real libtorch-vs-flex gradient vectors: old oracle
0 divergences, new oracle 3. Three of the four bugs here are special-value bugs,
so this was the harness blind to its own subject matter, and it explains why the
burn-flex `sign(NaN)` bug was found by reading code rather than by fuzzing.
`values_diverge` in `src/ir/interpreter/mod.rs` now branches on `is_nan` /
`is_infinite` first, with unit tests.

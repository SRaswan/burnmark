# Bugs

Active and past bugs found by this fuzzer, their root causes, fix locations,
and filing status. This is the working log — read it before touching any of
the five bugs below or resuming bug #4's investigation. See `CLAUDE.local.md`
for the project's goal and how to extend the fuzzer itself.

## Current state

Fuzzer targets Burn `0.22.0-pre.3` and runs clean. The original 0.20.1
`swap_dims` bug is confirmed fixed on 0.22.

**Two fixes are merged upstream.** Filing was the bottleneck; it no longer is
for those two.

| # | Bug | Owner repo | Status |
|---|---|---|---|
| 1 | aarch64 SIMD `recip()` silently loses ~0.2% precision on ≥32 elements (`vrecpeq_f32` with no Newton–Raphson refinement); corrupts `log()`'s gradient and `sigmoid()`'s forward pass | **macerator** (a burn *dependency*) | **pending** — branch `newton` in `~/Documents/Workspace/macerator`, 9 commits ahead of `upstream/main` |
| 2 | `sign(NaN)` returned ±1 from the raw sign *bit*, so `abs(log(x))`'s gradient was `1/x` garbage for `x < 0` | **burn-ndarray** *and* **burn-flex** | **MERGED** — [burn#5665](https://github.com/tracel-ai/burn/pull/5665), `66a8a5ff8`, 2026-09-15. One PR fixed both backends: ndarray's sign-bit read and flex's `copysign`-based `float_sign` (the flex half was first spotted independently by peer session `burn-f3`, commit `c11a8a037` in the shared checkout) |
| 3 | `powf_scalar(0.0)` returned a tensor detached from the autodiff graph (`float_powi_scalar`'s `0 => float_ones(...)` arm is a *constructor*, unrelated to `lhs`); every backend inherited it | **burn-backend** / autodiff | **MERGED** — [burn#5692](https://github.com/tracel-ai/burn/pull/5692), `98e48ddbd`, 2026-09-17, as "preserve gradients for zero scalar exponents" |
| 4 | `x.log().relu().relu()` with `x < 0`: gradient wrong in exactly the trailing `n mod 4` elements | burn-ndarray (suspected) | **open, not root-caused** — see below |
| 5 | `repeat_dim` backward groups incoming gradients as if the forward had *interleaved*, but the forward *tiles*. Silently wrong gradient whenever the repeated dim has size > 1 | **burn autodiff** (every backend) | **open, new** — found 2026-09-17, see below |

Per-bug root-cause traces and repros live in `docs/` —
`simd-recip-precision-bug.md`, `powi-scalar-zero-grad-bug.md`,
`relu-chain-nan-simd-remainder-bug.md` — with standalone reproductions in
`examples/`. Read the writeup before re-deriving anything.

### What's left to file

1. **#5 `repeat_dim` backward** — in burn's autodiff, so every backend inherits
   it, and nothing about it is deprecated. Same category strength as #3, which
   went in cleanly. Needs a repro in `examples/` and a root-cause line.
2. **#4 relu-chain NaN tail** — worth filing as an issue with the repro as-is
   even unrooted; the signature is precise enough that someone with
   checkpointing context could spot it fast. Note that others are actively
   fixing NaN bugs in this area right now ([#5658](https://github.com/tracel-ai/burn/pull/5658)
   flex relu/clamp NaN propagation, [#5662](https://github.com/tracel-ai/burn/pull/5662)
   flex max pooling, both by other contributors, both merged) — so this one is
   at real risk of being landed by someone else first.
3. **#1 macerator** — pending. It should ship with a linked burn-ndarray issue:
   the people who *hit* it are burn users who have no idea macerator exists.

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
burn's checkpointing internals.

**Cheapest next experiment:** a `simd`-off build. `burn-ndarray` and `burn-flex`
are both `default = ["std", "simd", "multi-threads"]`, so dropping `simd` yields
a scalar build of the identical backend. If the divergence vanishes, macerator's
tail handling is confirmed; if it persists, macerator is cleared and the
checkpointing hypothesis survives. Compile-time, so it is a two-build comparison
or a one-shot on the #4 repro — do the one-shot first. *Not yet run.*

### Newly found: bug #5, `repeat_dim` backward

Found by the raw tch-rs target within a minute of it being wired in, and it is
the kind of bug **no burn-vs-burn pairing could have found**: every burn backend
shares burn-autodiff, so they all compute the same wrong gradient and agree with
each other. libtorch's own autograd is the first independent second opinion this
fuzzer has ever had on a derivative.

Minimal repro — `x` is `[2×2]`, `y = sum(log(x.repeat_dim(0, 2)))`, so
`dy/dx == 2/x` exactly:

```
x = [0.225, 0.35, 0.475, 0.6]

forward   burn = [0.225, 0.35, 0.475, 0.6, 0.225, 0.35, 0.475, 0.6]   tile
      libtorch = [0.225, 0.35, 0.475, 0.6, 0.225, 0.35, 0.475, 0.6]   identical

dy/dx     burn = [6.549708, 4.5238094, 6.549708, 4.5238094]
      libtorch = [8.888889, 5.714286, 4.2105265, 3.3333333]           == 2/x, correct
```

burn's numbers are bit-exactly `[1/x₀+1/x₂, 1/x₁+1/x₃, 1/x₀+1/x₂, 1/x₁+1/x₃]` —
the grouping you get if the forward had been `repeat_interleave`. So the forward
tiles and the backward un-tiles as though it had interleaved. Equivalently, the
gradient is reshaped `[orig_dim, k, …]` and reduced over the wrong axis when it
should be `[k, orig_dim, …]` reduced over axis 0.

Reproduces identically on `ndarray` and `libtorch`, which places it in
burn-autodiff rather than a backend. Correct only when the repeated dimension
has size 1 (the two groupings coincide) — swept `[1×4] [2×2] [2×3] [3×2] [4×1]
[2×5] [3×3]` × `dim ∈ {0,1}` × `k ∈ {2,3}`; every case with
`size(dim) > 1` is wrong.

**Why burn's own tests miss it:** with a bare `sum(repeat_dim(x, d, k))` every
upstream gradient is `1`, so any mis-grouping of them sums to the same `k` and
the result is correct. It takes a *non-uniform* upstream gradient — the `log`
here — to expose the permutation. Any test that repeats and sums will pass.

Not yet root-caused to a line in burn's source, and no `examples/` repro
written yet.

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

`values_diverge` (`src/ir/interpreter/mod.rs`) used to decide divergence purely
by `abs_diff > 1e-4 * scale`, which is `false` whenever either side is `NaN` or
infinite — so **every** special-value divergence was silently reported as
agreement. Measured on the real libtorch-vs-flex gradient vectors: old oracle 0
divergences, new oracle 3. Most bugs here are special-value bugs, so this was
the harness blind to its own subject matter, and it explains why the burn-flex
`sign(NaN)` bug was found by reading code rather than by fuzzing. The rationale
now lives in that function's doc comment, with unit tests beside it.

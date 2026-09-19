# BurnMark

**Verifying and Benchmarking the Performance of Rust-Native LLM Agents vs Python**

This project evaluates the Rust ML ecosystem *correctness* via differential fuzzing:

- **Tensor-level differential fuzzer** that generates shape-aware SSA programs and tests Burn's autograd engine across backends — NdArray, burn-flex, LibTorch and CubeCL CPU, in any combination, plus two non-burn targets, raw tch-rs and candle — cross-checking both forward values and gradients. Found distinct autograd crashes in Burn 0.20.1 and five further distinct bugs in 0.22; two fixes are merged upstream ([#5665](https://github.com/tracel-ai/burn/pull/5665), [#5692](https://github.com/tracel-ai/burn/pull/5692)). Status for each lives in [`bugs.md`](bugs.md).

Backends:
- burn: NdArray, burn-flex, LibTorch, CubeCL CPU
- tch-rs
- candle

## Running the Fuzzer

All commands run from `burnmark/`. Requires `cargo-fuzz` (`cargo install cargo-fuzz`) and a nightly toolchain.

```bash
# Autograd fuzzing (backward-pass — where every bug so far has come from)
cargo +nightly fuzz run fuzz_autograd

# Multi-op tensor program fuzzing (forward-pass only)
cargo +nightly fuzz run fuzz_tensor_ops

# Run a specific crash artifact for reproduction
cargo +nightly fuzz run fuzz_autograd fuzz/artifacts/fuzz_autograd/<artifact-file>
```

### Choosing which backends to compare

Which backends a build *can* run is a compile-time feature; which it *does* run
is the `BACKENDS` environment variable, a comma-separated list. **The first entry
is the reference** — every other entry is compared against it, and divergences
are reported as `<name> vs <reference>`.

Most entries name a burn *device*. Two name different frameworks entirely:
`tch-raw` is libtorch called directly through tch-rs, and `candle` is candle
with neither burn nor libtorch anywhere in the path. `libtorch` and `tch-raw`
are deliberately distinct names for two different routes to the same C++
library — see below for why running both is the point.

```bash
# Build-time availability: ndarray is always in; these add the rest.
#   --features oracle-tch      → libtorch (burn's tch backend)
#   --features oracle-flex     → flex
#   --features oracle-cpu      → cpu (CubeCL CPU)
#   --features oracle-tch-raw  → tch-raw (raw tch-rs, no burn)
#   --features oracle-candle   → candle (candle-core, no burn and no libtorch)

# `cpu` needs -Cllvm-args=-asan-globals=0 (keeps ASAN; works around a linkme
# dupcheck false positive — see Roadmap) and a raised RSS limit, since CubeCL's
# JIT cache grows. Without both it dies in under a minute. `--sanitizer=none`
# also works but needlessly gives up the sanitizer.
RUSTFLAGS="-Cllvm-args=-asan-globals=0" \
BACKENDS=cpu,flex,ndarray  cargo +nightly fuzz run fuzz_autograd --features oracle-cpu,oracle-flex \
                             -- -rss_limit_mb=8192
RUSTFLAGS="-Cllvm-args=-asan-globals=0" \
BACKENDS=cpu,libtorch      cargo +nightly fuzz run fuzz_autograd --features oracle-cpu,oracle-tch \
                             -- -rss_limit_mb=8192

BACKENDS=tch-raw,libtorch  cargo +nightly fuzz run fuzz_autograd --features oracle-tch-raw,oracle-tch
BACKENDS=tch-raw,flex      cargo +nightly fuzz run fuzz_autograd --features oracle-tch-raw,oracle-flex
BACKENDS=candle,flex       cargo +nightly fuzz run fuzz_autograd --features oracle-candle,oracle-flex
# Two independent oracles against burn — see "triage by majority" below.
BACKENDS=tch-raw,candle,flex \
                           cargo +nightly fuzz run fuzz_autograd --features oracle-tch-raw,oracle-candle,oracle-flex
BACKENDS=libtorch,ndarray  cargo +nightly fuzz run fuzz_autograd --features oracle-tch
BACKENDS=libtorch,flex     cargo +nightly fuzz run fuzz_autograd --features oracle-tch,oracle-flex
BACKENDS=flex,ndarray      cargo +nightly fuzz run fuzz_autograd --features oracle-flex
BACKENDS=all               cargo +nightly fuzz run fuzz_autograd --features oracle-tch,oracle-flex,oracle-cpu,oracle-tch-raw,oracle-candle
BACKENDS=ndarray           cargo +nightly fuzz run fuzz_autograd   # no comparison; smoke/throughput runs
```

`BACKENDS=cpu,flex,ndarray` is the useful default once burn-tch is removed: it
keeps a reference believed correct without depending on a deprecated backend.
`BACKENDS=cpu,libtorch` is two independently implemented burn backends both
believed correct, so any disagreement between them is a finding rather than a
known-wrong backend restating a known bug.

`BACKENDS=tch-raw,libtorch` is the pairing that tests something none of the
others can, and it is worth being precise about what:

- **On the forward pass** both sides execute the *same* libtorch kernels, so
  there is no legitimate numerical difference between them. What differs is
  burn's FFI/translation layer, which means that layer is on its own under
  test. That is the class the original 0.20.1 `swap_dims` bug belonged to — a
  shallow clone across the FFI boundary corrupting gradients, not a math error —
  and no pairing of burn-internal backends can isolate it.
- **On the backward pass they are not the same implementation at all.**
  `Device::libtorch().autodiff()` differentiates with **burn-autodiff**;
  libtorch supplies only the tensors its formulas evaluate on. Every burn
  backend therefore shares one autograd, so no burn-vs-burn pairing — `cpu` vs
  `flex` vs `ndarray` vs `libtorch` — can disagree about a *derivative*, only
  about the forward kernels a derivative calls. `tch-raw` brings libtorch's own
  autograd, making this the only pairing here whose backward pass is written
  twice, independently. Since `fuzz_autograd` is where every bug so far has come
  from, that is the larger half of the case for it.

It is also the oracle burn cannot deprecate: `Device::libtorch()` is going away
on burn `main`, but `tch` is an independent project, and what burn is dropping
is the bridge, not the library.

That is not a theoretical argument. Within a minute of being wired in, this
pairing found **bug #5** — burn's `repeat_dim` tiles on the forward pass but
un-tiles the gradient as though it had interleaved, so the gradient lands on
the wrong elements whenever the repeated dimension has size > 1. Every burn
backend computes the same wrong answer and agrees with every other, which is
exactly why nothing before this could see it. Repro and root cause in
[`bugs.md`](bugs.md).

Two smaller backward divergences show up on published `0.22.0-pre.3` in
single-instruction programs: `x.powf(0)` (burn drops the node from the graph and
panics — bug #3, now fixed upstream but not in the published crate) and
`x.powf(-2)` at `x == 0` (burn `NaN`, libtorch `-inf`, at a genuinely singular
point). Neither is a translation artifact: the forward pass agrees on all 22
instructions, which `cargo test --features oracle-tch,oracle-tch-raw` asserts
exhaustively, isolated and chained.

`BACKENDS=tch-raw,candle,flex` is the pairing that makes a divergence
*triageable*, which is a different thing from finding one.

Every target other than `candle` shares an implementation with some other
target, and that bounds what a disagreement can mean. The four burn backends
share burn-autodiff, so they cannot disagree about a derivative. `tch-raw` and
`libtorch` share libtorch's forward kernels, so on the forward pass they cannot
disagree about the math. candle shares neither: its kernels are its own and so
is its autograd, making it the first target here that is independent on **both**
passes at once. Two things follow, and the second is the one that earns its
keep:

- A `candle` vs `libtorch` forward agreement is *evidence* — two unrelated
  implementations landing on the same number. `tch-raw` vs `libtorch` agreement
  is close to a tautology.
- **Triage by majority.** With two independent oracles running against burn,
  where both agree and burn does not, burn is wrong and the report is a finding
  ready to minimize; where the two oracles split, the question is about candle
  rather than about burn, and the report can be set aside cheaply. Every
  previous pairing produced reports that had to be root-caused before anyone
  could tell which side was wrong.

The cost is real and worth stating: a candle-vs-burn divergence does **not**
localise the bug the way `libtorch` vs `tch-raw` does. Those two differ by
exactly one thing — burn's FFI bridge — so a divergence there names its own
culprit. candle and burn differ by two entire implementations. Running candle
alongside `tch-raw` rather than instead of it is what buys the localisation
back.

candle is also a second answer to the deprecation problem. `Device::libtorch()`
is going away on burn `main`; `tch` survives that because it is an independent
project, and candle survives it twice over, sharing not even a C++ library with
anything burn ships.

**Two known candle divergences, found within 90 s of wiring it in.** Both are
candle's own, not burn's — on each, `libtorch`, `flex` and `ndarray` all agree
*against* candle:

| case | candle | libtorch / flex / ndarray |
|---|---|---|
| `relu'(0)` | `1.0` | `0.0` |
| `d/dx log(relu(x))` at `x < 0` | `NaN` | `0.0` |

Both come from candle's relu backward (`backprop.rs:634`), which masks with
`ge(0)` where PyTorch's `threshold_backward` uses strictly-greater, and
*multiplies* by that 0/1 mask where PyTorch *selects* — so `-inf × 0` becomes
`NaN` instead of `0`. Since `relu` on negative inputs is common, a
`fuzz_autograd` campaign including candle will halt on these repeatedly; run
candle as a **third** target so the majority vote localises it, or use it in
`fuzz_tensor_ops` (forward only), where it is clean. Note what *not* to do:
special-casing `relu` in `values_diverge` would also hide bug #4, which is an
open NaN-in-chained-`relu` bug in exactly this neighbourhood.

Two translation caveats are worth knowing, both in
[`ir/interpreter/candle.rs`](src/ir/interpreter/candle.rs)'s module doc: candle's
plain `add`/`sub`/`mul`/`div` do not broadcast (the `broadcast_*` forms are the
ones with burn's semantics), and `sigmoid` lives in `candle-nn` rather than
`candle-core` — it is pulled in for that one op rather than open-coding
`(1 + (-x).exp()).recip()`, since an open-coded version would differ from burn
in *composition* and those differences are indistinguishable from the bugs this
fuzzer looks for.

Unset `BACKENDS` runs everything compiled in, reference side first — `tch-raw`
when `oracle-tch-raw` is compiled in, else `candle`, else `cpu`, else
`libtorch`, else `ndarray` alone. A selection naming an unknown entry, or one this build lacks,
fails loudly with the missing feature named rather than silently comparing fewer
sides than asked for.

Crash artifacts are stored in `fuzz/artifacts/fuzz_autograd/`. Example reproductions are in `examples/`.

---

## Roadmap

Four distinct bugs have come out of fuzzing Burn `0.22.0-pre.3` so far — an
aarch64 SIMD `recip()` precision loss in the [`macerator`](https://github.com/wingertge/macerator)
dependency ([writeup](docs/simd-recip-precision-bug.md)), a `sign(NaN)` gradient
corruption in `burn-ndarray` (and independently in `burn-flex`), a
`powf_scalar(0.0)` autodiff-graph detachment ([writeup](docs/powi-scalar-zero-grad-bug.md)),
and a still-unrooted NaN/SIMD-remainder divergence in chained `relu`
([writeup](docs/relu-chain-nan-simd-remainder-bug.md)).

Every one of them was found the same way: patch the *previous* bug's fix into
the fuzzer's dependency graph, re-run, see what surfaces next. That workflow is
what the roadmap below is really about automating.

### Phase 0 — deepen the generator

The fuzzer's reach is currently bounded in four specific places:

- **Type-directed generation.** [`AutogradProgram`](src/ir/generate.rs) has a
  shape-aware state-machine builder that only ever emits legal ops. The plain
  `TensorProgram` path does not — its `Vec<TensorInstr>` comes straight from
  `#[derive(Arbitrary)]`, and the interpreter fixes up illegal operands after
  the fact (`raw % num_regs`, then `resolve_broadcast_compatible` /
  `resolve_matmul_compatible` fallbacks). That wastes generation budget and
  distorts the op distribution: a program's tail is mostly whatever the fixups
  chose, not what the mutator asked for. `MAX_TENSOR_ELEMENTS` in
  [`src/ir/shape.rs`](src/ir/shape.rs) is a band-aid over the same gap — it
  exists only because unguided `Repeat`/`Concat`/`Matmul` chains could compound
  into a multi-gigabyte allocation. Extend the builder to both program types and
  pick each op from the set that is legal *given the current arena*, so nothing
  is generated and then thrown away.
- **Higher-rank tensors.** [`Shape2`](src/ir/shape.rs) is hardcoded 2-D and is
  the single source of truth for all shape algebra; the interpreter is
  `Tensor<2>` throughout. Burn's `Tensor<B, D>` is const-generic in rank, so
  supporting rank 1–5 means a rank-erased dispatch enum over `D` plus a dynamic
  `ShapeN` for the generator's arena. This matters beyond coverage-for-its-own-sake:
  rank is where `swap_dims`, `flip`, `narrow`, per-dim reductions and broadcast
  edge cases live, and the original 0.20.1 bug was a `swap_dims` bug.
- **Op coverage.** Current set is in [`src/ir/ops.rs`](src/ir/ops.rs). Highest-value
  additions, roughly in order of how much backend-specific code they reach:
  `swap_dims`/`permute`, `reshape`, `slice`/`slice_assign`, `gather`/`select`,
  `mask_fill`/`mask_where`, `softmax`/`log_softmax`, `min`/`max`/`clamp_min`/`clamp_max`,
  tensor–tensor `powf`, `recip` directly (rather than only reaching it through
  `log`/`sigmoid`), `var`/`std`, `prod`, `cumsum`, `sort`/`argmax`, `stack`/`chunk`,
  `tril`/`triu`, `erf`, `sin`/`cos`.
- **Special-value seeding.** `bytes_to_floats` maps seed bytes uniformly into
  `[-1, 1]`, so it never *directly* produces `NaN`, `±inf`, subnormals, signed
  zero, or exact `±1` boundaries — those only arise incidentally, downstream of
  something like `log(x < 0)`. Three of the five bugs found so far are
  special-value dependent. A seeded special-value pool mixed into leaf data is
  probably the cheapest available yield increase in the whole roadmap.

Three supporting changes decide whether a longer run is actually a *better*
run, rather than just a bigger wall-clock number:

- **Campaign discipline.** `-max_total_time` / `-jobs N` for parallel runs,
  `cargo fuzz cmin` to keep [`fuzz/corpus`](fuzz/corpus/) from bloating, and
  `cargo fuzz coverage` to report which ops and which backend code paths are
  actually being exercised — so op-coverage work is driven by a coverage gap
  rather than by guessing.
- **~~Backend coverage.~~ Done.** Both targets now run any combination of
  NdArray, burn-flex, LibTorch, CubeCL CPU and raw tch-rs, selected at runtime.
  See [Choosing which backends to compare](#choosing-which-backends-to-compare);
  what each one cost to add is below.
- **Oracle tolerance is itself a bug-hiding knob.** The `recip` precision bug
  was a ~0.2% relative error that `macerator`'s own test suite missed because
  its tolerance was `2^-8`. `compare_outputs`'s tolerance should be tracked
  deliberately, kept tight, and paired with an explicit, documented allowlist
  for ops where last-bit transcendental divergence between backends is genuinely
  expected — rather than one loose global epsilon that quietly absorbs real bugs.
- **~~The oracle cannot see any special-value divergence~~ — fixed.** The old
  comparison decided divergence solely with `abs_diff > 1e-4 * scale`, which is
  `false` for every pair involving a `NaN` or an infinity — so the harness was
  blind to exactly the bug class most of its findings belong to.
  `values_diverge` now branches on `is_nan` / `is_infinite` first, with unit
  tests for each case. Measured consequence and the bugs it cost: see
  [`bugs.md`](bugs.md); the reasoning lives in that function's doc comment.

#### Adding a backend

Burn 0.22 dropped the `Backend` type parameter from `Tensor` — which backend
runs an op is a property of the `Device`, not of the tensor's type — so
[`collect_grads`](src/ir/interpreter/autograd.rs) and
[`eval_tensor_program`](src/ir/interpreter/tensor_program.rs) already take a
plain `&Device` and are backend-generic as written. Measured, not assumed: one
cargo feature (`burn/metal`) plus one `Device::metal(DeviceKind::DefaultDevice)`
call ran the full autograd path on the GPU with **zero changes to any
interpreter file**, and `.autodiff()` is a `Device` method, so backward-pass
fuzzing comes along with it.

Every constructor available in burn-tensor 0.22.0-pre.3, each behind a burn
cargo feature: `flex()`, `cpu()`, `cuda(i)`, `rocm(i)`, `wgpu(kind)`,
`vulkan(kind)`, `metal(kind)`, `webgpu(kind)`, `libtorch_cuda(i)`,
`libtorch_mps()`, `libtorch_vulkan()`. The CPU-side ones — `flex()`, `cpu()`,
`libtorch_mps()` — really are a feature flag and a line each.

The work is therefore not in the interpreter. It is in three places:

1. **~~The oracle is hardcoded 2-way.~~ Done.** It now takes an ordered backend
   list with the first entry as reference, reports *which* backends disagree and
   at how many elements, and names every diverging backend rather than stopping
   at the first. Selection is runtime (`BACKENDS`), so one binary covers
   `libtorch,ndarray`, `libtorch,flex`, `flex,ndarray`, or all three — see
   [Choosing which backends to compare](#choosing-which-backends-to-compare).
2. **GPU panics escape `catch_as_result`, silently — fix this first.** The Metal
   probe hit two shader-compilation failures (`thread 'DSD-4-0' panicked ...
   Failed to generate the backend-specific code`) on a device *runner* thread.
   `catch_as_result` uses `catch_unwind`, which only catches panics on the
   calling thread, so the process **exited 0 with two kernels failed and still
   printed a numeric answer**. For a differential fuzzer whose whole premise is
   trusting the numbers, "a kernel failed, we got a result anyway, exit 0" is
   strictly worse than crashing — it manufactures false confidence. Needs a
   global panic hook that records cross-thread panics into a flag the harness
   checks before it compares anything. (Those particular shader failures may be
   driver-specific; the escape path is not.)
3. **Device construction cost and readback sync.** ~0.31 s per process, almost
   entirely GPU device creation. `Device::ndarray()` is currently built *inside*
   each `run_*` call — free at CPU cost, fatal under libFuzzer at GPU cost — so
   it needs hoisting to a `OnceLock`. Every program also ends in `into_data()`,
   forcing a GPU→CPU readback per iteration; that is the throughput ceiling and
   should be measured before a GPU goes anywhere near the hot loop.

Two more things to expect. The first is **no longer a prediction**: `cargo-fuzz`
defaults to ASAN, and every CubeCL backend does trip over it — but not for the
guessed reason, and the remedy is narrower than expected. It is not GPU driver
false positives; it is `linkme`'s dupcheck misfiring under ASAN inside CubeCL's
`pliron` compiler layer, which means it bites the **CPU** backend too, with no
GPU anywhere. `--sanitizer=none` works but overshoots: dropping just ASAN's
global redzones with `-Cllvm-args=-asan-globals=0` keeps the sanitizer. Measured
detail under "The CubeCL CPU backend is now wired in" below. The second is still
untested: the `1e-4` relative tolerance in `compare_outputs` is probably too
tight for GPU transcendentals (`exp`/`log`/`tanh`/`sigmoid`) against LibTorch
CPU, so expect false-positive noise until tolerance is per-op.

**Easiest backend to add right now: `flex()`.** Measured against the
alternatives on this machine:

| | flex | cpu (CubeCL) | metal / wgpu | libtorch_mps | cuda / rocm |
|---|---|---|---|---|---|
| New dependencies | `burn/flex` only | cubecl, no driver stack | cubecl + wgpu + driver stack | **none** (`tch` already on) | — |
| Cold build | **14 s** | 1 m 08 s | minutes | none | — |
| Background-thread panics | **0** | **0** | 2 shader failures | 0 | — |
| Needs `OnceLock` / readback / panic hook | **no** | **no** | all three | no | — |
| Runs under ASAN | **yes** | **no** (linkme dupcheck) | untested | yes | — |
| exec/s (30 s, `-max_len=8`) | 5,334 (ASAN: 1,834) | 323 (ASAN: 227) | untested | untested | — |
| RSS over a run | untested | climbs (260→562 Mb) | untested | untested | — |
| Differential value | **high** (replaces NdArray) | **high** (CubeCL kernels, correct on `sign(NaN)`) | high (3rd implementation) | low (PyTorch vs PyTorch) | high |
| Works here | yes | yes | with caveats | yes | no hardware |

Because burn-flex is pure-Rust CPU and synchronous, none of the three costs
above apply to it: device construction is trivial, `into_data()` is not a device
sync, and there are no worker threads to panic on. It is the one backend that
can be added without touching `catch_as_result` or hoisting device construction
— and it is the non-deprecated replacement for the backend three of the five
known bugs live in or reach through. **It is now wired in**, behind `--features oracle-flex`.

That deprecation cuts deeper than it first appears: on burn `main`,
`Device::libtorch()` is deprecated too ("burn-tch is deprecated and will be
removed in a future release. Use a CubeCL backend ... or `Device::flex()`
instead"), though not yet in published `0.22.0-pre.3`. So **both sides of the
original oracle are on their way out** — NdArray and LibTorch alike — and the
backends that survive are burn-flex and the CubeCL family. That makes adding a
CubeCL backend a matter of keeping a trustworthy reference at all, not just
breadth: LibTorch is the best reference available *today*, and that is
time-limited.

**The CubeCL CPU backend is now wired in**, behind `--features oracle-cpu`, and
it takes the reference slot ahead of LibTorch — it is the only backend that is
both measured correct on the `sign(NaN)` case and not deprecated. None of the
three GPU prerequisites above apply to it: no cross-thread panics to catch, no
device construction to hoist, no per-op tolerance needed.

It has three costs of its own, though, and all three were measured only by
actually running it — none was predicted:

1. **It breaks under *stock* ASAN** — fixable with one flag, see the end of this
   item. CubeCL reaches `pliron` (its MLIR-style compiler layer)
   via `cubecl-cpu` → `cubecl-llvm`, and `pliron` registers dictionary keys with
   `linkme`'s `#[distributed_slice]`. Under ASAN that panics immediately:

   ```
   duplicate #[distributed_slice] with name "DICT_KEY_IDS"
   ```

   This is **not** a burn, CubeCL or pliron bug — it is a `linkme`/ASAN
   incompatibility, reproducible in a crate with none of them present:

   ```rust
   #[distributed_slice] pub static THINGS: [&'static str];
   ```
   ```
   cargo run                                  → THINGS = ["a", "b"]
   RUSTFLAGS="-Zsanitizer=address" cargo run  → duplicate #[distributed_slice]
   ```

   ASAN pads globals with redzones, which widens the gap between `linkme`'s
   dupcheck sentinels until its `dupcheck_start + 1 < dupcheck_stop` test fires
   spuriously (`linkme-0.3.37/src/distributed_slice.rs:231`). (`-Clink-dead-code`,
   cargo-fuzz's other default flag and the more obvious suspect, was tested and is
   *not* the cause.)

   **The fix keeps ASAN**: `RUSTFLAGS="-Cllvm-args=-asan-globals=0"` drops only
   ASAN's *global* redzones. Verified on the real build — `BACKENDS=cpu` runs
   clean, ASAN is still linked (116 `__asan` symbols plus the asan runtime
   dylib), and a minimal crate confirms it still catches heap-buffer-overflows.
   Only *global*-buffer-overflow detection is lost, which barely matters when
   every tensor is heap-allocated. Measured price: `cpu` 227 exec/s under ASAN vs
   323 with `--sanitizer=none` — about 1.4×.

2. **CubeCL JIT-compiles kernels at runtime, so it is ~16× slower and its RSS
   climbs.** Measured over 30 s on identical settings (`-max_len=8`):

   | | ndarray | cpu |
   |---|---|---|
   | exec/s | **5,334** | 323 (degrading: 682 → 409 → 323) |
   | total execs | 165,361 | 11,317 |
   | RSS | 97 Mb, flat | 260 → 474 → 562 Mb, climbing |

   The climb is why libFuzzer's default `-rss_limit_mb=2048` kills a `cpu` run
   within a minute; raise it. Whether RSS plateaus or grows without bound is
   **not** yet measured — 11k execs was not long enough to tell.

3. **Its coverage signal is dominated by the compiler, not by burn.** The same
   30 s produced 4,726 coverage points on ndarray and **39,898** on cpu — 8.4×
   more, because the instrumented JIT is itself being executed. That is not more
   tensor-op coverage; it means libFuzzer's corpus selection will optimise toward
   inputs that stress CubeCL's compiler rather than burn's math. Treat `cpu`
   coverage numbers as incomparable to the other backends'.

`metal()` remains the next one after, once the cross-thread panic hook exists —
and it will inherit costs 1 and 3, since it shares this CubeCL compiler path.

The probe was worth running on its own account. Gradient of `abs(log(x))` over
`[-0.5, 0.25, 2.0, -3.0]`, against unpatched published `0.22.0-pre.3`:

| Backend | Gradient | |
|---|---|---|
| `libtorch` | `[-0.0, -4.0, 0.5, -0.0]` | correct — `sign(NaN) == 0` |
| `libtorch_mps` | `[-0.0, -4.0, 0.5, -0.0]` | correct |
| `metal` (CubeCL) | `[-0.0, -4.0, 0.5, -0.0]` | correct |
| `cpu` (CubeCL) | `[-0.0, -4.0, 0.5, -0.0]` | correct — re-measured after wiring it in |
| `ndarray` | `[-2.0, -4.0, 0.5, -0.33333334]` | wrong: `1/x`, sign taken from the NaN's sign *bit* |
| `flex` | `[NaN, -4.0, 0.5, NaN]` | wrong: returns the NaN itself |

Three conclusions, from a twenty-line program with no fuzzer involved:

- `burn-cubecl`'s `sign` — the one backend left unaudited for this bug class —
  looks **correct** on the Metal path. One input, one op, one shader compiler is
  not an audit, but it is evidence CubeCL is a usable third oracle rather than a
  fourth suspect.
- burn-flex's `sign(NaN)` bug is present in the **published crate**, not only on
  `main`, which is worth stating in that fix's PR.
- Two backends are wrong in two *different* ways, which is the argument for the
  N-way oracle refactor over simply swapping the reference side: a two-way
  comparison cannot express this, and whichever backend were designated the
  reference, the other's failure would be misread.

### Phase 1 — automatic crash characterization

A raw crash artifact is not yet a filable bug. Everything between "libFuzzer
saved a file" and "here is a one-line repro and the exact condition that
triggers it" is currently manual, and it is the same handful of moves every
time. Worth building as a harness in its own right:

- **Minimize**, beyond `cargo fuzz tmin`: delta-debug at the *IR* level (drop
  SSA instructions, shrink leaf shapes, collapse register aliasing) rather than
  at the byte level, since the IR is what the writeup needs to quote.
- **Cluster by root cause, not by artifact.** A single unfixed bug saturates the
  crash channel: the `recip` bug produced 261 crashes in one 5-minute
  continuous-mode run, all one root cause; `powf_scalar(0.0)` produced 640–647
  divergent runs per 25-minute session, also all one root cause. Deduping on a
  signature (panic message + minimized op sequence + divergence shape) is what
  turns a pile of artifacts into a triage queue.
- **Sweep the parameter space around a crash automatically.** The `relu`-chain
  bug's characterization — needs `NaN`, needs *two* chained `relu`s, needs
  `n mod 4 != 0` where 4 is the NEON `f32` lane width — came from manually
  sweeping `n` from 16 to 257 and individually confirming each condition was
  necessary. That's mechanical: given a minimized repro, sweep element count,
  rank, op-repetition count, and NaN-vs-finite inputs, and report which
  conditions are load-bearing. The `recip` bug had the same "≥ 32 elements"
  shape, so this sweep is reusable, not one-off.
- **Bisect the regression.** The `sign(NaN)` bug was traced to burn PR #4573
  ("move sign back to mathOps"), where a missing `PartialOrd` bound caused
  NaN-safe comparison logic to be rewritten as `is_positive()`. `git log -S` on
  the implicated function plus `git bisect` against the repro is scriptable, and
  naming the causing commit is most of what makes a report credible to a
  maintainer.

### Phase 2 — the fix-and-continue loop

The headline idea: an agent watches the fuzzer, and when it crashes, triages it,
fixes the *actual target repo*, commits to an isolated branch, re-points the
fuzzer at the patched tree, and resumes — so the campaign keeps going past each
bug instead of rediscovering it forever.

The motivation is not novelty, it's throughput. Because one unfixed bug
saturates the crash channel (see the 261-in-5-minutes number above), the fuzzer
cannot find bug *N+1* until bug *N* is patched in. Fix latency *is* discovery
rate. Every distinct bug found in 0.22 so far was found only after hand-patching
the previous one.

Most of the loop's primitives already exist here, as manual steps:

| Loop step | What already does this by hand |
|---|---|
| Point the fuzzer at patched code | `[patch.crates-io]` in [`fuzz/Cargo.toml`](fuzz/Cargo.toml), currently aimed at two local fixed checkouts |
| Isolate each fix | a per-bug `git worktree` (`burn-powi-fix`), used precisely because a peer session held the shared checkout on another branch |
| Run with several fixes at once | `fix-powi-scalar-zero-grad` branched on top of `fix-sign-nan` — the stack that Phase 3 replaces with a regenerable integration branch |
| Pin a dependency whose API drifted | a worktree off the `macerator-v0.3.4` tag with the fix cherry-picked, because `main` had renamed `Arch::new()` → `Arch::detect()` |
| Keep repros honest | root `Cargo.toml` deliberately *not* patched, so `examples/` still demonstrate the bug against published crates.io versions |

So the loop is mostly orchestration over moves that are already proven, not new
machinery:

1. **Run** until a crash or divergence; capture the artifact.
2. **Characterize** via Phase 1 (minimize → dedup → sweep → bisect). If the
   signature matches an already-patched bug, discard and continue — *before*
   spending anything on triage (see Phase 4).
3. **Root-cause** to a specific line in the target or a dependency. This is the
   step that must not be skipped: the three bugs that reached a PR did so only
   because the root cause was pinned to a line (`sign_op`'s `is_positive()`,
   `float_powi_scalar`'s `float_ones`, `recip_f32`'s missing Newton–Raphson
   refinement) — and bug #4 is still unfiled precisely
   because it isn't.
4. **Patch** in a fresh worktree on a fresh branch — one bug per branch, based
   on current upstream `main`, *not* stacked on the previous fix (Phase 3).
5. **Validate** before accepting the fix — a gate, not a formality. The
   de-facto protocol that all three existing fixes went through:
   - the target's own test suite (e.g. `burn-backend-tests` `--test tensor` +
     `--test autodiff`), with zero new failures and zero new ignores;
   - a new regression test that provably *fails before and passes after* — and
     if the upstream test's tolerance was what hid the bug, tighten it (the
     `recip` fix pulled `macerator`'s check from `2^-8` to `2^-20`);
   - the standalone minimal repro re-run against the patched tree, now matching
     the oracle exactly;
   - **replay of the original crash artifact** against a rebuilt fuzz target —
     the only check that proves the fuzzer will actually get past this bug.
6. **Falsify the fix.** Revert it in a scratch worktree and confirm the original
   artifact crashes *again*. If it doesn't, the fix was never load-bearing and
   the real bug is still out there. This step exists because the loop's worst
   failure mode isn't a missing fix, it's a **wrong fix that silences the signal**:
   a patch that suppresses the crash without addressing the cause makes the
   fuzzer stop finding that entire bug class, and the resulting quiet reads
   exactly like coverage. Cheap, mechanical, and the only thing keeping
   auto-accept honest.
7. **Re-point and continue**: regenerate the integration branch, rewrite
   `fuzz/Cargo.toml`'s `[patch.crates-io]`, rebuild, and go to 1 — now hunting
   whatever this bug was masking.
8. **Draft** the submission: writeup into `docs/`, minimal repro into
   `examples/`, and the exact `gh` invocation, staged and ready.

Step 8 is where the loop stops. It does not file anything (see the human gate
below).

### Phase 3 — branch topology on a fork

All agent branches live on a **fork** of the target, never on upstream. The
agent keeps fixing and pushing; promoting a fix to a real upstream PR is a
separate, human decision. That promotion is cheap — a branch already pushed to
the fork becomes an upstream PR with one command, no rework:

```bash
gh pr create --repo tracel-ai/burn --head <you>:fix-sign-nan-ndarray --base main
```

**The topology matters, because two consumers of these branches want opposite
things.** The fuzzer needs *every* fix present simultaneously — that is the
whole unmasking mechanism. An upstream reviewer needs each fix *alone*, based on
upstream `main`: no burn maintainer will take one PR containing an ndarray NaN
fix, a flex NaN fix, and an unrelated `powf_scalar` autodiff fix.

A linear stack can't satisfy both, and this one did bite:
`fix-powi-scalar-zero-grad` sat on top of `fix-sign-nan`, so promoting the powi
fix alone meant unpicking it from a fix it had nothing to do with — under
exactly the time pressure this project exists to avoid. Both eventually went
upstream as separate PRs ([#5665](https://github.com/tracel-ai/burn/pull/5665),
[#5692](https://github.com/tracel-ai/burn/pull/5692)), which is the shape below,
arrived at by hand. The point of Phase 3 is to not arrive at it by hand.

So: **independent siblings, plus one throwaway integration branch.**

```
fix-sign-nan-ndarray      ─┐   each: one bug, one commit, based on origin/main,
fix-sign-nan-flex         ─┤   independently promotable to an upstream PR
fix-powi-scalar-zero-grad ─┤
                           └─→ fuzz-integration   ← what fuzz/Cargo.toml points at
```

```bash
git checkout -B fuzz-integration origin/main
git cherry-pick <tip of each fix branch>    # regenerate from scratch; never commit onto it
```

This also serves the "easier to make code changes and rebase" goal *better* than
a stack does: every branch is a single commit off `main`, so keeping up with
upstream is `git rebase main` per branch with no restacking, and integration
conflicts get resolved once per regeneration instead of accumulating in history.
These fixes touch different crates and files, so cherry-pick conflicts are
unlikely in practice.

Stacking still earns its place where fixes are *genuinely* dependent — same
function, or B's regression test needs A — and that's where
[Graphite](https://graphite.dev) or `gh pr create --base <parent-branch>` chains
pay off. The point is only that the fuzzer's need for cumulative fixes should
not be the reason for a stack: the integration branch covers that. Note too that
fixes here already span two repos (`macerator` and `burn`), which a single stack
cannot express at all.

Two more things the fork model buys, worth designing for rather than
discovering by accident:

- **Fork CI is free cross-architecture validation.** Every bug found so far is
  aarch64-SIMD-flavoured, and the `macerator` fix is literally
  `#[cfg(target_arch = "aarch64")]` code. There is no way to verify on this
  machine that it doesn't regress x86 — but pushing to the fork runs the
  target's own workflows on hardware nobody here owns. That's an argument for
  pushing early and often, independent of filing anything.
- **Cross-repo linking.** Bugs don't respect crate boundaries: the `recip` fix
  belongs in `macerator`, but the people who *hit* it are `burn-ndarray` users
  who have no idea `macerator` exists. A dependency-side PR should ship with a
  linked issue on the consumer, and every writeup should name which repo owns
  the fix and which owns the symptom.

Alongside the branches, a **campaign ledger**: one row per distinct bug —
signature, repro, root-cause line, branch, validation status, filed/unfiled.
Mostly so "unfiled" is a visible, uncomfortable number rather than something
that quietly sits.

### Phase 4 — the orchestration stack

The governing principle: **deterministic harness, agent only at the judgment
steps.** Running the fuzzer, watching for crashes, hashing signatures,
regenerating branches, rewriting `[patch.crates-io]`, rebuilding — all ordinary
code, no tokens, no nondeterminism. An agent is invoked for root-cause, patch,
regression test, and writeup. Nothing is gained by having a model babysit a
fuzzing loop.

**[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk) as the spine.**
The SDK is Python/TypeScript only and this project is Rust, so the orchestrator
is a sidecar that shells out to `cargo fuzz` (or drives the CLI headless with
`-p --output-format json` from any language). What matters for an unattended
multi-hour campaign:

- **Dedup before invoking, never after.** One bug produced 261 crashes in five
  minutes; paying for a full triage on each would be absurd for a single root
  cause. The signature hash is cheap code, and only a first-of-signature crash
  should ever reach a model.
- **A fresh session per crash, with a pre-assembled context bundle** — minimized
  IR, sweep table, bisect result, the implicated source file. The SDK's own
  session docs recommend exactly this over transcript resumption: capture what
  you need as application state and pass it into a new session. Reserve
  `resume` for continuing one specific triage.
- **Hard spend and concurrency caps**: `max_budget_usd` / `maxBudgetUsd` (which
  surfaces as an `error_max_budget_usd` result rather than a surprise bill),
  plus `CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS` and
  `CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH`.
- **File checkpointing** for clean rejection — sessions persist the
  conversation, not the filesystem, so this is the primitive that reverts a fix
  the validation gate turned down.

**Skills** are the highest value per hour of work here, because the procedures
are already checklist-shaped and already written down in prose: `triage-crash`
(minimize → dedup → sweep → bisect), `validation-gate` (the protocol in Phase 2
step 5, plus the falsification in step 6), `write-bug-report` (the `docs/`
template now used three times), `patch-and-repoint`. The payoff is that the SDK
loads skills and memory from `.claude/` the same way the interactive CLI does
(gated by `setting_sources`), so the unattended loop and a hands-on session run
the *same* procedure from one source of truth — fix the skill and both improve.

**Hooks** are what turn the non-goals below from prose into enforcement: a
`PreToolUse` hook that refuses `git push --force` and writes into the shared
checkout, a `PostToolUse` hook for `cargo fmt` and clippy. A rule an agent is
merely *asked* to follow is not a guarantee; a blocked tool call is.

**Subagents** for two specific jobs: parallel triage of distinct signatures in
separate worktrees, and — more valuable — a **falsifier** whose only task is to
break the patch another agent just wrote (Phase 2 step 6). The author of a fix
is the worst reviewer of it, and subagent context isolation means the falsifier
doesn't inherit the reasoning that produced the patch. One gotcha: only the
Agent tool's prompt string crosses from parent to subagent, so artifact and file
paths have to be passed explicitly.

**MCP, mostly skipped.** A third-party GitHub MCP server buys nothing that `gh`
doesn't already do (including `gh search issues` to check whether a bug is
already reported upstream); adding one would be cargo-culting. The exception is
the *in-process* SDK server — `@tool` + `create_sdk_mcp_server` — wrapping this
project's own harness operations as typed tools: `minimize_artifact`,
`sweep_param`, `replay_artifact`, `run_validation_gate`. Better than having a
model reconstruct long `cargo +nightly fuzz` incantations with the right
`LIBTORCH` / `DYLD_LIBRARY_PATH` every time.

**Cheapest starting point:** prototype the loop interactively — a couple of the
skills above, driven on an interval — before writing any orchestrator code.
Building the sidecar first risks automating a workflow that isn't right yet.

### Prior art, and what would actually be new

Worth stating carefully, since "first of its kind" is the sort of claim people
check:

- **OSS-Fuzz / ClusterFuzz** already do continuous fuzzing, crash bucketing,
  automatic bisection, filing, and verify-and-close. They don't generate fixes.
- **DARPA AIxCC** (finals, Aug 2025) scored Cyber Reasoning Systems on finding
  *and* automatically patching vulnerabilities, with validation. This is the
  closest prior art and it is substantial.
- **Google's LLM-based patching of OSS-Fuzz findings** generates candidate fixes
  with human review before submission.
- **Differential testing of DL frameworks** is a whole literature — CRADLE,
  LEMON, Muffin, EAGLE, NNSmith, Tzer — including typed graph generation and
  gradient checking.

So "fuzzer finds bug, agent patches it, human files it" is not new. Three things
about *this* setup do look unusual:

- **The oracle is a second implementation, not a sanitizer.** Nearly all
  auto-patch work targets crashes, UB, or vulnerabilities, where a sanitizer
  declares the bug. Four of the five bugs here produce no crash at all — 0.2%
  wrong numbers, a wrong `-0.0`, a silently-`None` gradient, a gradient
  permuted onto the wrong elements. There is nothing to
  bucket on, and the comparison tolerance is itself a knob that can hide bugs.
- **Patch-to-unmask instead of suppress.** The standard answer to "one bug
  drowns the channel" is suppression or bucketing, which works fine for a stack
  trace. It does not work for `recip()` being 0.2% wrong: you cannot suppress
  that without also suppressing `log`, `sigmoid`, and everything composed from
  them. In this domain, *fixing* is the only way to keep fuzzing — which is a
  far better justification for the loop than novelty.
- **Cross-repo patching across a dependency boundary** — fix in `macerator`,
  symptom in `burn-ndarray`, injected via `[patch.crates-io]`. AIxCC-style tasks
  are single-repo.

The defensible version of the claim, then: *an agentic differential-fuzzing loop
where each validated fix is injected back into the dependency graph to unmask
the next bug, applied to numerical and autodiff correctness rather than memory
safety.* Narrow, true, and still interesting. Check the current literature
before putting the word "first" in writing anywhere.

### What to file first

**Live status for every bug is in [`bugs.md`](bugs.md)** — what is merged, what
is pending, and what is still unfiled. Two fixes are upstream
([#5665](https://github.com/tracel-ai/burn/pull/5665) `sign(NaN)` across
burn-ndarray *and* burn-flex, [#5692](https://github.com/tracel-ai/burn/pull/5692)
autodiff gradients for zero scalar exponents); the macerator `recip` refinement
is pending; two bugs remain unfiled.

The principle that ordering followed, and still applies: `Device::ndarray()`
carries `#[deprecated(since = "0.22.0")]`, so a fix to burn-ndarray alone is
worth less to a maintainer than the same bug class fixed in burn-flex (its
replacement) or in `burn-backend`/autodiff (which every backend inherits). Both
merged PRs were of the latter two kinds. Bug #5 (`repeat_dim` backward) is in
autodiff, which is why it leads the unfiled list.

That deprecation also settled the reference-side question recorded in
[`tensor_program.rs`](src/ir/interpreter/tensor_program.rs): NdArray is no
longer the reference. The order now lives in `Target::ALL`
([`src/ir/program.rs`](src/ir/program.rs)) — raw tch-rs first, then CubeCL CPU,
then LibTorch, then the two backends known to be wrong on `sign(NaN)`.

### The human gate, and non-goals

The loop deliberately stops one step short of submitting. Branch pushed, tests
green, writeup drafted, `gh` command printed — then it waits. Filing is a
person's call, every time: the loop cannot judge whether a fix is the *right*
fix for a maintainer's codebase, and an auto-filed PR from a fuzzing bot is a
worse artifact than the same patch filed by someone who can defend it in review.

What the loop is *for* is making sure "unfiled" never means "forgotten": the
cost of a found bug should be a few minutes of review, not a context reload
weeks later.

Explicit non-goals: no auto-merging, no PRs opened against the upstream repo
(the fork is the agent's whole world), no force-pushing to shared branches (a
peer session was working in the shared checkout — hence the worktree
discipline), no editing another session's branch, and no silently patching the
root `Cargo.toml`, which would "fix" the bug reproductions in `examples/` and
destroy their entire purpose. These belong in hooks, not just in this list.

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | HuggingFace API token for gated models |
| `LLM_BENCH_SECTION3_ONLY=1` | Skip Sections 1 & 2, run only Section 3 |
| `LLM_BENCH_TRAIN_TRANSPOSE_ONLY=1` | Train only the transpose-tied model variant |
| `LLM_BENCH_CANDLE_MODEL=<key>` | Select Candle model without `--model` flag |
| `CANDLE_GGUF_PATH` / `CANDLE_TOKENIZER_PATH` | Override model/tokenizer paths with local files |

## Use of AI

Parts of this codebase were developed with assistance from Claude (Anthropic). AI was used for code generation, debugging, architectural planning, and report writing. All AI-generated code was reviewed, tested, and validated by the team.

# BurnMark

**Differential fuzzer for Rust ML / autograd crates.**

Generates shape-aware SSA tensor programs, runs them across multiple backends and frameworks simultaneously, and flags any divergence in forward values or gradients. Found and upstreamed five distinct bugs in Burn 0.22 ([#5665](https://github.com/tracel-ai/burn/pull/5665), [#5692](https://github.com/tracel-ai/burn/pull/5692)); full bug log in [`bugs.md`](bugs.md).

## Architecture

The codebase splits into two sharp layers:

| Layer | Files | Burn imports |
|---|---|---|
| **IR + generation** — framework-independent | `src/ir/ops.rs`, `ir/program.rs`, `ir/generate.rs`, `ir/shape.rs`, `ir/interpreter/shape.rs` | 0 |
| **Driver** — shared SSA walk, leaf/alias protocol | `ir/interpreter/driver.rs` | 0 (generic over `Framework`) |
| **Burn interpreter** | `ir/interpreter/mod.rs`, `autograd.rs`, `tensor_program.rs` | minimal |
| **tch-rs interpreter** (`--features oracle-tch-raw`) | `ir/interpreter/tch_raw.rs` | 0 |
| **candle interpreter** (`--features oracle-candle`) | `ir/interpreter/candle.rs` | 0 |

The IR (ops, shapes, programs) is entirely framework-agnostic. Each interpreter is an independent ~22-arm match over the same `TensorInstr` enum — adding a new target means a new interpreter file and two one-line wrappers, not a trait or a generic.

**There is exactly one trait**, `Framework` in `driver.rs`, and it abstracts the *scaffolding around* the match (register file, leaf seeding, grad extraction) — never the 22 arms themselves. See [`CLAUDE.local.md`](CLAUDE.local.md) for the full architecture notes before extending.

## Backends / Targets

| Name | Framework | Feature flag |
|---|---|---|
| `ndarray` | Burn (always available) | — |
| `flex` | Burn | `oracle-flex` |
| `libtorch` | Burn (deprecated in 0.22 main) | `oracle-tch` |
| `cpu` | Burn / CubeCL CPU | `oracle-cpu` |
| `tch-raw` | tch-rs direct (no burn) | `oracle-tch-raw` |
| `candle` | candle-core (no burn, no libtorch) | `oracle-candle` |

`libtorch` and `tch-raw` are two routes to the same C++ library — running both isolates burn's FFI bridge. `candle` shares nothing with any other target, making `tch-raw,candle,flex` the most informative triple: where both oracles agree and burn doesn't, burn is wrong.

## Running

Requires `cargo-fuzz` and a nightly toolchain. LibTorch targets need `LIBTORCH` and `DYLD_LIBRARY_PATH` set.

```bash
# Autograd fuzzing (backward pass — where all bugs so far came from)
cargo +nightly fuzz run fuzz_autograd

# Forward-pass multi-op fuzzing
cargo +nightly fuzz run fuzz_tensor_ops

# Replay a crash artifact
cargo +nightly fuzz run fuzz_autograd fuzz/artifacts/fuzz_autograd/<file>
```

`BACKENDS` selects which targets run at runtime (first entry = reference):

```bash
BACKENDS=tch-raw,candle,flex  cargo +nightly fuzz run fuzz_autograd \
  --features oracle-tch-raw,oracle-candle,oracle-flex

BACKENDS=tch-raw,libtorch     cargo +nightly fuzz run fuzz_autograd \
  --features oracle-tch-raw,oracle-tch

BACKENDS=flex,ndarray         cargo +nightly fuzz run fuzz_autograd \
  --features oracle-flex

BACKENDS=all                  cargo +nightly fuzz run fuzz_autograd \
  --features oracle-tch,oracle-flex,oracle-cpu,oracle-tch-raw,oracle-candle
```

**CubeCL CPU (`cpu`) needs two extra flags** — its JIT compiler trips a `linkme`/ASAN false positive and has high RSS:

```bash
RUSTFLAGS="-Cllvm-args=-asan-globals=0" BACKENDS=cpu,flex \
  cargo +nightly fuzz run fuzz_autograd --features oracle-cpu,oracle-flex \
  -- -rss_limit_mb=8192
```

## Vision — self-healing fuzz loop

The longer-term goal is a loop where finding a bug automatically leads to fixing it: crash → IR-level minimization → root cause → patch in an isolated worktree → validation gate → branch pushed to fork. The loop stops short of filing — that's a human call — but it gets to "ready to submit" without manual steps.

Each fix lives on its own branch off upstream `main` (independently promotable as a PR) and is cherry-picked onto a throwaway integration branch that the fuzzer points at, so it can keep running past each bug and find the next one. Fixes can span dependency boundaries (`macerator` fix injected into `burn-ndarray` via `[patch.crates-io]`).

This is the "self-healing code" angle: differential fuzzing continuous enough to unmask bugs in layers, with patches re-injected automatically so no single bug saturates the channel.

Full roadmap, orchestration design, and prior-art comparison: [`plans.md`](plans.md).

## Key files

| File | Purpose |
|---|---|
| `src/ir/ops.rs` | `TensorInstr` enum — the instruction vocabulary |
| `src/ir/generate.rs` | Shape-aware SSA program generator |
| `src/ir/shape.rs` | `Shape2`, all shape algebra, allocation cap |
| `src/ir/interpreter/driver.rs` | Shared SSA walk (`Framework` trait) |
| `src/ir/interpreter/mod.rs` | Burn interpreter + `values_diverge` |
| `fuzz/fuzz_autograd.rs` | Backward-pass fuzz target |
| `fuzz/fuzz_tensor_ops.rs` | Forward-pass fuzz target |
| `bugs.md` | All found bugs, root causes, filing status |
| `plans.md` | Roadmap: generator, crash characterization, fix loop, orchestration |

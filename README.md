# fuzzburn

fuzzburn generates random tensor programs and runs them through Burn, looking for:

| Bug class | How it is detected |
|---|---|
| Panics / assertion failures | libFuzzer catches any `panic!` / `abort` |
| Missing gradients | `fuzz_autograd` asserts both leaf tensors have `Some` gradient after `backward()` |
| Shape / dtype errors | Triggered by the op-sequence structure driving illegal combinations |

I want **differential fuzzing** in the future, so we run the same generated program through two backends (e.g. NdArray vs WGPU) and compare numeric outputs.

---

# Repository Structure

This repository is organized as a Cargo workspace with two main components:

## fuzzburn (root crate)
Fuzzing infrastructure for Burn tensor operations

## llm_benchmark
LLM inference benchmarking tool for comparing Burn backends and Candle performance

---

## Architecture

The fuzzer is built around a small **IR / AST**:

```
arbitrary (libFuzzer bytes)
      │
      ▼
 TensorProgram / AutogradProgram   ← program.rs  (AST root)
      │  owns a Vec of
      ▼
 TensorOp / DiffOp / TensorRef    ← ops.rs       (instruction set)
      │
      ▼
 interpreter.rs                   ← tree-walker → Burn tensor calls
```

This separation means:
- AST: op enums and program structs are **Burn-agnostic** – they describe *what* to compute.
- Burn entry: `interpreter.rs` is the only file that imports Burn – swapping backends or adding a second backend for differential testing only requires changes there.

## File structure (gpt gen lol)

```
fuzzburn/
├── Cargo.toml                  # main crate (fuzzburn lib + bin)
├── llm_benchmark/              # LLM benchmarking subcrate
│   ├── Cargo.toml
│   ├── src/
│   └── README.md
├── src/
│   ├── main.rs                 # stub entry point / usage hints
│   ├── lib.rs                  # re-exports `pub mod ir`
│   └── ir/
│       ├── mod.rs              # module declarations
│       ├── ops.rs              # TensorOp, DiffOp, TensorRef enums
│       ├── program.rs          # TensorProgram, AutogradProgram (AST roots)
│       └── interpreter.rs      # evaluates a program against Burn
├── examples/
│   └── simple_autograd.rs      # minimal y = x² autograd demo
└── fuzz/
    ├── Cargo.toml              # cargo-fuzz crate, depends on fuzzburn
    ├── fuzz_tensor_ops.rs      # fuzz target: plain tensor API
    └── fuzz_autograd.rs        # fuzz target: autodiff + backward pass
```

---

# Running llm_benchmark

The `llm_benchmark` crate provides LLM inference benchmarking capabilities. It can benchmark:
1. **Burn backend comparison** – Custom GPT architecture across NdArray (CPU), WGPU (GPU), and optionally LibTorch (PyTorch)
2. **Candle pure-Rust generation** – Real quantised GGUF models via HuggingFace Candle
3. **Training benchmark** – Burn's TUI dashboard for training steps

## Basic llm_benchmark commands

```bash
# Run basic Burn backend benchmarks (no download required)
cargo run --release -p llm_benchmark

# Run with Candle support (downloads ~700 MB GGUF on first run)
cargo run --release -p llm_benchmark --features candle

# macOS Apple Silicon – use Metal GPU for Candle
cargo run --release -p llm_benchmark --features candle,metal

# With training + TUI dashboard
cargo run --release -p llm_benchmark --features train

# All features (Candle + Metal + training)
cargo run --release -p llm_benchmark --features candle,metal,train

# Include LibTorch/PyTorch backend (requires libtorch installed)
cargo run --release -p llm_benchmark --features tch
```

For more details about llm_benchmark configuration and options, see `llm_benchmark/README.md`.

---

# Running fuzzburn

## Prerequisites

```sh
# Rust nightly is required by cargo-fuzz / libFuzzer
rustup install nightly

# Install cargo-fuzz
cargo install cargo-fuzz
```

## Running the examples

```sh
cargo run --example simple_autograd
```

## Running the fuzz targets

```sh
# Fuzz the plain tensor API (shape ops, activations, reductions)
cargo +nightly fuzz run fuzz_tensor_ops

# Fuzz the autodiff backend (backward pass, gradient correctness)
cargo +nightly fuzz run fuzz_autograd

# Fuzz autodiff with LibTorch oracle comparison
cargo +nightly fuzz run fuzz_autograd --features oracle-tch
```

Useful flags:

```sh
# Limit each run to 1 second of wall time (good for CI)
cargo fuzz run fuzz_tensor_ops -- -max_total_time=60

# Run with more parallelism
cargo fuzz run fuzz_autograd -- -workers=4

# Minimise a crashing input after finding a bug
cargo fuzz tmin fuzz_autograd <path/to/crash>
```

Crash artifacts are saved to `fuzz/artifacts/<target>/`.

---

## Roadmap

- [ ] **burn-ir differential fuzzing** – lower the same `TensorProgram` AST into `burn_ir::OperationDescription` nodes and replay it on two backends, comparing outputs.
- [ ] Higher-rank tensors (3-D, 4-D) and batched ops.
- [ ] Structured seed corpus of known interesting inputs.

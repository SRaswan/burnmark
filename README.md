# BurnMark

**Verifying and Benchmarking the Performance of Rust-Native LLM Agents**

This project evaluates the Rust ML ecosystem across two axes: *correctness* via differential fuzzing and *performance* via benchmarking against PyTorch. It comprises two components:

- **Tensor-level differential fuzzer/** that generates shape-aware SSA programs and tests Burn's autograd engine across NdArray, WGPU, and LibTorch backends. Discovered 17 distinct autograd crashes in Burn 0.20.1.
- **Three-section benchmarking suite/** comparing Burn and Candle (pure-Rust LLM inference) against PyTorch on forward-pass throughput, real-model inference, and training workloads.


## Running the Benchmark (`llm_benchmark/`)

All commands run from `llm_benchmark/`.

### Quick Start

```bash
# All three sections (Candle defaults to TinyLlama ~670 MB download on first run)
cargo run --release  -p llm_benchmark --features candle,train -- --model tinyllama
```

### Selecting Benchmark Sections

| Command | Sections Run |
|---------|-------------|
| `cargo run --release -p llm_benchmark` | 1 only |
| `cargo run --release -p llm_benchmark --features candle` | 1 + 2 |
| `cargo run --release -p llm_benchmark --features train` | 1 + 3 |
| `cargo run --release -p llm_benchmark --features candle,train` | 1 + 2 + 3 |
| `LLM_BENCH_SECTION3_ONLY=1 cargo run --release -p llm_benchmark --features train` | 3 only |

- **Section 1** — Burn backend throughput (NdArray CPU vs WGPU GPU, random weights)
- **Section 2** — Candle LLM inference (real GGUF model, greedy decoding) — requires `candle` feature
- **Section 3** — Training throughput with Burn Autodiff + TUI dashboard — requires `train` feature

### Other Useful Commands

##### GPU Acceleration 

```bash
# macOS Apple Silicon (Metal) — for Candle Section 2
cargo run --release -p llm_benchmark --features candle,metal,train

# CUDA
cargo run --release -p llm_benchmark --features candle,cuda,train

# LibTorch backend in Section 1 (requires libtorch installed)
cargo run --release -p llm_benchmark --features tch
```

##### Model Selection for Section 2

```bash
# List available models
cargo run --release -p llm_benchmark --features candle -- --list-models

# Select a model
cargo run --release -p llm_benchmark --features candle -- --model phi3

# Gated models require a HuggingFace token
HF_TOKEN=hf_xxx cargo run --release -p llm_benchmark --features candle -- --model llama3-1b
```

| Key | Model | Size | Token Required |
|-----|-------|------|----------------|
| `tinyllama` | TinyLlama-1.1B Q4_K_M | ~670 MB | No |
| `phi3` | Phi-3-Mini-4K Q4 | ~2.3 GB | No |
| `llama3-1b` | Llama-3.2-1B Q4_K_M | ~0.8 GB | Yes |
| `llama3-3b` | Llama-3.2-3B Q4_K_M | ~2.0 GB | Yes |

##### Training Variants for Section 3

```bash
# Transpose-tied model only (exercises the autograd path where the fuzzer found crashes)
LLM_BENCH_TRAIN_TRANSPOSE_ONLY=1 cargo run --release -p llm_benchmark --features train
```

##### Python Baseline

```bash
python python_benchmark/benchmark.py
```

Runs the same GPT architecture in PyTorch with identical configs for Section 1 comparison. Auto-selects MPS/CUDA/CPU.

---

## Running the Fuzzer

All commands run from `burnmark/`. Requires `cargo-fuzz` (`cargo install cargo-fuzz`) and a nightly toolchain.

```bash
# Autograd fuzzing (backward-pass, found the 17 crashes)
cargo +nightly fuzz run fuzz_autograd

# Multi-op tensor program fuzzing (forward-pass only)
cargo +nightly fuzz run fuzz_tensor_ops

# Run a specific crash artifact for reproduction
cargo +nightly fuzz run fuzz_autograd fuzz/artifacts/fuzz_autograd/<artifact-file>
```

Crash artifacts are stored in `fuzz/artifacts/fuzz_autograd/`. Example reproductions are in `examples/`.

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

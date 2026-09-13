# BurnMark

**Verifying and Benchmarking the Performance of Rust-Native LLM Agents vs Python**

This project evaluates the Rust ML ecosystem *correctness* via differential fuzzing:

- **Tensor-level differential fuzzer** that generates shape-aware SSA programs and tests Burn's autograd engine across NdArray, WGPU, and LibTorch backends. Discovered 17 distinct autograd crashes in Burn 0.20.1.

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

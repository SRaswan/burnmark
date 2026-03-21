# llm_benchmark

Benchmarks LLM inference performance in Rust across two independent sections:

1. **Burn backend comparison** – identical custom GPT architecture run against NdArray (CPU), WGPU (GPU), and optionally LibTorch (PyTorch). Measures forward-pass throughput (input tokens/sec) with random weights so only the backend/hardware differs.

2. **Candle pure-Rust generation** – loads a real quantised GGUF model (TinyLlama Q4_K_M by default) via HuggingFace Candle and runs autoregressive generation on a fixed prompt set. Measures TTFT, output tokens/sec, and per-token latency p50/p95.

---

## Running

```bash
# Section 1 only (Burn backends, no download required)
cargo run --release

# Section 1 + Section 2 (Candle, downloads ~700 MB GGUF on first run)
cargo run --release --features candle

# macOS Apple Silicon – use Metal GPU for Candle
cargo run --release --features candle,metal

# Section 1 + Section 3 (training + TUI dashboard)
cargo run --release --features train

# Section 2 + Section 3 (Candle + training)
cargo run --release --features candle,train

# All three sections (Candle, Metal, and training)
cargo run --release --features candle,metal,train

# Include LibTorch/PyTorch backend in section 1 (requires libtorch installed)
cargo run --release --features tch
```

Model files are cached by `hf-hub` in `~/.cache/huggingface/hub/` after the first download.

## Apples-to-apples methodology

**Section 1 (Burn)**
- Same `GptConfig`, same random input shape, same warm-up + iteration count across all backends. Only the Burn backend (and thus the compute kernel) changes.
- Metric: input tokens processed per second (forward-pass throughput).

**Section 2 (Candle)**
- Same GGUF weights, same tokeniser, same prompts (`src/prompts.rs`), greedy decoding (temperature = 0, no sampling randomness).
- Model load time is measured separately and excluded from all throughput numbers.
- Metrics: TTFT (prefill latency), output tokens/sec (decode throughput), per-token latency p50/p95, peak memory usage (RSS).

**Memory benchmarking**
- Both sections now include peak memory usage measurement during benchmark execution.
- Uses process RSS (Resident Set Size) from `/proc/self/statm` on Linux.
- Memory usage is reported in MB alongside other metrics.

## Project structure

```
src/
  main.rs           – orchestrates both benchmark sections
  model.rs          – GPT-style decoder-only model (Burn)
  benchmark.rs      – BenchmarkStats (forward-pass) + GenerationStats (generation)
  candle_runner.rs  – Candle GGUF loader and greedy decode loop (feature: candle)
  prompts.rs        – fixed prompt set used by section 2
Cargo.toml
```

## Burn model configs

| Config | vocab | hidden | layers | heads | seq |
|--------|------:|-------:|-------:|------:|----:|
| tiny   |   512 |    128 |      2 |     2 |  64 |
| small  |  2048 |    256 |      4 |     4 | 128 |

To add a custom size, add a method to `GptConfig` in `src/model.rs` and pass it to `benchmark_model` in `main.rs`.

## Benchmark settings

Defaults in `BenchmarkConfig` (`src/benchmark.rs`): batch size 4, sequence length 32, 50 iterations. Preset alternatives: `BenchmarkConfig::quick()` (20 iters) and `BenchmarkConfig::thorough()` (100 iters).

## Candle model

Default: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` · `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`  
To switch models, change `CandleModelConfig` in `src/candle_runner.rs` – any GGUF-format LLaMA-architecture model works.

use crate::benchmark::{benchmark_model, compare_results, BenchmarkConfig};
#[cfg(feature = "candle")]
use crate::benchmark::GenerationStats;
use crate::config::{CliConfig, RuntimeConfig};
use crate::model::{Gpt, GptConfig};
use burn_ndarray::{NdArray, NdArrayDevice};
#[cfg(feature = "tch")]
use burn_tch::{LibTorch, LibTorchDevice};
use burn_wgpu::{Wgpu, WgpuDevice};

pub fn model_configs() -> Vec<(&'static str, GptConfig)> {
    vec![("tiny", GptConfig::tiny()), ("small", GptConfig::small())]
}

pub fn run_section1(model_configs: &[(&str, GptConfig)], bench_config: &BenchmarkConfig) {
    println!("{:=<80}", "");
    println!(" SECTION 1: Burn Backend Throughput (identical architecture, random weights)");
    println!("{:=<80}\n", "");
    println!(
        "Apples-to-apples: same GptConfig, same random input tensor shape,\n\
         same iteration count, same warm-up passes. Only backend differs.\n\
         Metric: input tokens processed per second (forward-pass throughput).\n"
    );

    let mut burn_results = Vec::new();

    for (model_name, gpt_config) in model_configs {
        println!("\n{:->60}", "");
        println!(
            " Model: {}  (vocab={}, hidden={}, layers={}, heads={})",
            model_name.to_uppercase(),
            gpt_config.vocab_size,
            gpt_config.hidden_size,
            gpt_config.num_layers,
            gpt_config.num_heads
        );
        println!("{:->60}\n", "");

        println!("[ NdArray (CPU) ]");
        let device_ndarray = NdArrayDevice::Cpu;
        let model_ndarray = Gpt::<NdArray>::new(gpt_config, &device_ndarray);
        let r = benchmark_model(
            &model_ndarray,
            gpt_config,
            bench_config.batch_size,
            bench_config.sequence_length,
            bench_config.num_iterations,
            &device_ndarray,
            "NdArray (CPU)",
            model_name,
        );
        r.print_summary();
        burn_results.push(r);

        println!("[ WGPU (GPU) ]");
        let device_wgpu = WgpuDevice::default();
        let model_wgpu = Gpt::<Wgpu>::new(gpt_config, &device_wgpu);
        let r = benchmark_model(
            &model_wgpu,
            gpt_config,
            bench_config.batch_size,
            bench_config.sequence_length,
            bench_config.num_iterations,
            &device_wgpu,
            "WGPU (GPU)",
            model_name,
        );
        r.print_summary();
        burn_results.push(r);

        #[cfg(feature = "tch")]
        {
            println!("[ LibTorch (PyTorch CPU) ]");
            let device_tch = LibTorchDevice::Cpu;
            let model_tch = Gpt::<LibTorch<f32>>::new(gpt_config, &device_tch);
            let r = benchmark_model(
                &model_tch,
                gpt_config,
                bench_config.batch_size,
                bench_config.sequence_length,
                bench_config.num_iterations,
                &device_tch,
                "LibTorch (PyTorch CPU)",
                model_name,
            );
            r.print_summary();
            burn_results.push(r);
        }
    }

    compare_results(&burn_results);
}

pub fn run_section2(cli: &CliConfig, runtime: &RuntimeConfig) {
    #[cfg(feature = "candle")]
    {
        use crate::benchmark::compare_generation_results;
        use crate::candle_runner::{print_available_models, resolve_model, CandleRunner};
        use crate::prompts::{BENCHMARK_PROMPTS, WARMUP_PROMPT};

        if !cli.section3_only {
            println!("\n{:=<80}", "");
            println!(" SECTION 2: Candle Pure-Rust LLM (real weights, real prompts)");
            println!("{:=<80}\n", "");
            println!(
                "Apples-to-apples: same prompts, same max_new_tokens,\n\
                 greedy decoding (temperature=0). Model load time is excluded.\n\
                 Metrics: TTFT (prefill latency), output tokens/sec (decode throughput),\n\
                 per-token latency p50/p95.\n"
            );

            let model_key = cli.model_key.as_deref().unwrap_or("tinyllama");

            let model_config = match resolve_model(model_key) {
                Some(config) => Some(config),
                None => {
                    eprintln!("  Error: unknown model key '{}'\n", model_key);
                    print_available_models();
                    eprintln!("  Skipping Section 2.\n");
                    None
                }
            };

            if let Some(model_config) = model_config {
                if let Some(token) = runtime.hf_token.as_deref() {
                    if std::env::var("HF_TOKEN").is_err()
                        && std::env::var("HUGGING_FACE_HUB_TOKEN").is_err()
                    {
                        std::env::set_var("HF_TOKEN", token);
                    }
                }

                println!(
                    "  Selected model: {} (key: '{}')",
                    model_config.display_name, model_key
                );
                println!("  Loading model: {} ...\n", model_config.display_name);

                match CandleRunner::load(model_config.clone()) {
                    Err(e) => {
                        eprintln!(
                            "  [Candle] Could not load model - skipping section 2.\n  Reason: {e}\n\
                             Tip: ensure you have internet access for the first run (model is cached\n\
                             afterwards). You can also set HF_TOKEN if the repo is gated.\n"
                        );
                    }
                    Ok(mut runner) => {
                        println!(
                            "  Model loaded in {} ms (excluded from throughput numbers)\n",
                            runner.model_load_time_ms
                        );

                        println!("  Warming up with '{}' prompt ...", WARMUP_PROMPT.name);
                        let _ = runner.generate(&WARMUP_PROMPT);
                        println!();

                        let mut gen_results = Vec::new();

                        for prompt in BENCHMARK_PROMPTS {
                            println!(
                                "  Prompt '{}': {} chars, max {} new tokens",
                                prompt.name,
                                prompt.text.len(),
                                prompt.max_new_tokens
                            );

                            match runner.generate(prompt) {
                                Ok(stats) => {
                                    stats.print_summary();
                                    gen_results.push(stats);
                                }
                                Err(e) => {
                                    eprintln!("  [Candle] Generation error on '{}': {e}", prompt.name)
                                }
                            }
                        }

                        compare_generation_results(&gen_results);
                        summarize_model_run(model_config.display_name, &gen_results);
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "candle"))]
    {
        let _ = (cli, runtime);
        println!("\n{:=<80}", "");
        println!(" SECTION 2: Candle Pure-Rust LLM - DISABLED");
        println!("{:=<80}", "");
        println!(
            "\n  Enable with: cargo run --features candle\n\
             On macOS with Apple Silicon add --features candle,metal for GPU.\n"
        );
    }
}

pub fn run_section3(
    model_configs: &[(&str, GptConfig)],
    bench_config: &BenchmarkConfig,
) {
    #[cfg(feature = "train")]
    {
        use burn::backend::Autodiff;

        use crate::training::run_training_benchmark;

        let train_transpose_only = std::env::var("LLM_BENCH_TRAIN_TRANSPOSE_ONLY")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);

        println!("\n{:=<80}", "");
        println!(" SECTION 3: Training Throughput + Burn TUI Dashboard");
        println!("{:=<80}\n", "");
        println!(
            "Trains the same GPT model with Adam (random data, cross-entropy loss)\n\
             across NdArray-Autodiff (CPU) and WGPU-Autodiff (GPU).\n\
             The terminal will switch to the ratatui TUI for each backend -\n\
             it shows a live loss curve, items/sec, and epoch progress.\n\
             Tip: set LLM_BENCH_TRAIN_TRANSPOSE_ONLY=1 to run only the transpose-tied model.\n"
        );

        let train_epochs = 3;
        let train_batch = bench_config.batch_size;

        for (model_name, gpt_config) in model_configs {
            let train_label = if train_transpose_only {
                format!("{}-TRANSPOSE", model_name.to_uppercase())
            } else {
                model_name.to_uppercase()
            };
            println!("{:->60}", "");
            println!(" Model: {}  (training)", train_label);
            println!("{:->60}", "");

            if train_transpose_only {
                #[cfg(feature = "tch")]
                run_training_benchmark::<
                    Autodiff<LibTorch<f32>>,
                    crate::model::GptTranspose<Autodiff<LibTorch<f32>>>,
                    _,
                >(
                    gpt_config,
                    LibTorchDevice::Cpu,
                    "LibTorch-Autodiff (CPU)",
                    if gpt_config.hidden_size <= 128 {
                        "tiny-transpose"
                    } else {
                        "small-transpose"
                    },
                    train_epochs,
                    train_batch,
                    crate::model::GptTranspose::<Autodiff<LibTorch<f32>>>::new,
                );

                run_training_benchmark::<
                    Autodiff<NdArray>,
                    crate::model::GptTranspose<Autodiff<NdArray>>,
                    _,
                >(
                    gpt_config,
                    NdArrayDevice::Cpu,
                    "NdArray-Autodiff (CPU)",
                    if gpt_config.hidden_size <= 128 {
                        "tiny-transpose"
                    } else {
                        "small-transpose"
                    },
                    train_epochs,
                    train_batch,
                    crate::model::GptTranspose::<Autodiff<NdArray>>::new,
                );

                run_training_benchmark::<
                    Autodiff<Wgpu>,
                    crate::model::GptTranspose<Autodiff<Wgpu>>,
                    _,
                >(
                    gpt_config,
                    WgpuDevice::default(),
                    "WGPU-Autodiff (GPU)",
                    if gpt_config.hidden_size <= 128 {
                        "tiny-transpose"
                    } else {
                        "small-transpose"
                    },
                    train_epochs,
                    train_batch,
                    crate::model::GptTranspose::<Autodiff<Wgpu>>::new,
                );
            } else {
                #[cfg(feature = "tch")]
                run_training_benchmark::<Autodiff<LibTorch<f32>>, Gpt<Autodiff<LibTorch<f32>>>, _>(
                    gpt_config,
                    LibTorchDevice::Cpu,
                    "LibTorch-Autodiff (CPU)",
                    if gpt_config.hidden_size <= 128 { "tiny" } else { "small" },
                    train_epochs,
                    train_batch,
                    Gpt::<Autodiff<LibTorch<f32>>>::new,
                );

                run_training_benchmark::<Autodiff<NdArray>, Gpt<Autodiff<NdArray>>, _>(
                    gpt_config,
                    NdArrayDevice::Cpu,
                    "NdArray-Autodiff (CPU)",
                    if gpt_config.hidden_size <= 128 { "tiny" } else { "small" },
                    train_epochs,
                    train_batch,
                    Gpt::<Autodiff<NdArray>>::new,
                );

                run_training_benchmark::<Autodiff<Wgpu>, Gpt<Autodiff<Wgpu>>, _>(
                    gpt_config,
                    WgpuDevice::default(),
                    "WGPU-Autodiff (GPU)",
                    if gpt_config.hidden_size <= 128 { "tiny" } else { "small" },
                    train_epochs,
                    train_batch,
                    Gpt::<Autodiff<Wgpu>>::new,
                );
            }
        }
    }

    #[cfg(not(feature = "train"))]
    {
        let _ = (model_configs, bench_config);
        println!("\n{:=<80}", "");
        println!(" SECTION 3: Training Benchmark - DISABLED");
        println!("{:=<80}", "");
        println!(
            "\n  Enable with: cargo run --features train\n\
             Runs full forward+backward+Adam steps with the Burn TUI dashboard.\n"
        );
    }
}

#[cfg(feature = "candle")]
fn summarize_model_run(model_name: &str, results: &[GenerationStats]) {
    if results.is_empty() {
        println!("\n  No successful generations recorded for {model_name}.\n");
        return;
    }

    let avg_ttft_ms = results
        .iter()
        .map(|r| r.ttft.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / results.len() as f64;

    let avg_out_toks_per_sec = results
        .iter()
        .map(GenerationStats::output_tokens_per_sec)
        .sum::<f64>()
        / results.len() as f64;

    println!("\n  Model summary: {model_name}");
    println!("  - prompts completed: {}", results.len());
    println!("  - average TTFT: {:.1} ms", avg_ttft_ms);
    println!("  - average decode throughput: {:.2} output tok/s\n", avg_out_toks_per_sec);
}

use std::time::{Duration, Instant};
use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use rand::Rng;
use sysinfo::{ProcessRefreshKind, System, get_current_pid};

use crate::model::{Gpt, GptConfig};

#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    pub rss_bytes: u64,
    pub vsz_bytes: u64,
}

impl MemoryStats {
    pub fn current() -> Result<Self, Box<dyn std::error::Error>> {
        let pid = get_current_pid()
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        let mut system = System::new();
        system.refresh_process_specifics(pid, ProcessRefreshKind::new().with_memory());
        let process = system
            .process(pid)
            .ok_or("Could not find current process")?;
        Ok(MemoryStats {
            rss_bytes: process.memory(),
            vsz_bytes: process.virtual_memory(),
        })
    }

    pub fn rss_mb(&self) -> f64 {
        self.rss_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn rss_gb(&self) -> f64 {
        self.rss_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

pub fn measure_memory_usage<F, T>(f: F) -> Result<(T, MemoryStats), Box<dyn std::error::Error>>
where
    F: FnOnce() -> T,
{
    let before = MemoryStats::current()?;
    let result = f();
    let after = MemoryStats::current()?;
    
    let peak_memory = MemoryStats {
        rss_bytes: after.rss_bytes.saturating_sub(before.rss_bytes),
        vsz_bytes: after.vsz_bytes.saturating_sub(before.vsz_bytes),
    };
    
    Ok((result, peak_memory))
}

#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub prompt_name: String,
    pub backend_name: String,
    pub prompt_token_count: usize,
    pub output_token_count: usize,
    pub ttft: Duration,
    pub decode_time: Duration,
    pub per_token_latencies_us: Vec<u64>,
    pub p50_latency_us: u64,
    pub p95_latency_us: u64,
    pub peak_memory: MemoryStats,
}

impl GenerationStats {
    pub fn new(
        prompt_name: &str,
        backend_name: &str,
        prompt_token_count: usize,
        output_token_count: usize,
        ttft: Duration,
        decode_time: Duration,
        peak_memory: MemoryStats,
    ) -> Self {
        Self {
            prompt_name: prompt_name.to_string(),
            backend_name: backend_name.to_string(),
            prompt_token_count,
            output_token_count,
            ttft,
            decode_time,
            per_token_latencies_us: Vec::new(),
            p50_latency_us: 0,
            p95_latency_us: 0,
            peak_memory,
        }
    }

    pub fn output_tokens_per_sec(&self) -> f64 {
        if self.decode_time.is_zero() || self.output_token_count == 0 {
            return 0.0;
        }
        self.output_token_count as f64 / self.decode_time.as_secs_f64()
    }

    pub fn total_time(&self) -> Duration {
        self.ttft + self.decode_time
    }

    pub fn compute_percentiles(&mut self) {
        if self.per_token_latencies_us.is_empty() {
            return;
        }
        let mut sorted = self.per_token_latencies_us.clone();
        sorted.sort_unstable();
        let n = sorted.len();
        self.p50_latency_us = sorted[n / 2];
        self.p95_latency_us = sorted[(n as f64 * 0.95) as usize];
    }

    pub fn print_summary(&self) {
        println!(
            "\n  ┌── Generation: {} │ {} ──",
            self.prompt_name, self.backend_name
        );
        println!(
            "  │  Prompt tokens:         {:<10}",
            self.prompt_token_count
        );
        println!(
            "  │  Output tokens:         {:<10}",
            self.output_token_count
        );
        println!(
            "  │  Time To First Token:   {:.2?}",
            self.ttft
        );
        println!(
            "  │  Decode throughput:     {:.2} output tok/s",
            self.output_tokens_per_sec()
        );
        println!(
            "  │  Total time:            {:.2?}",
            self.total_time()
        );
        if !self.per_token_latencies_us.is_empty() {
            println!(
                "  │  Per-token latency p50: {} µs",
                self.p50_latency_us
            );
            println!(
                "  │  Per-token latency p95: {} µs",
                self.p95_latency_us
            );
        }
        println!(
            "  │  Peak memory (RSS):     {:.2} MB",
            self.peak_memory.rss_mb()
        );
        println!("  └─────────────────────────────────────────");
    }
}

pub fn compare_generation_results(results: &[GenerationStats]) {
    if results.is_empty() {
        return;
    }

    println!("\n╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           GENERATION BENCHMARK COMPARISON                                  ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:<28} │ {:<26} │ {:>8} │ {:>9} │ {:>9} │ {:>8} ║",
        "Backend", "Prompt", "TTFT ms", "out tok/s", "p50 µs", "Mem MB"
    );
    println!("╟─────────────────────────────┼───────────────────────────┼─────────┼──────────┼──────────┼─────────╢");

    for r in results {
        println!(
            "║ {:<28} │ {:<26} │ {:>7.1} │ {:>8.1} │ {:>8} │ {:>7.1} ║",
            truncate(&r.backend_name, 28),
            truncate(&r.prompt_name, 26),
            r.ttft.as_secs_f64() * 1000.0,
            r.output_tokens_per_sec(),
            r.p50_latency_us,
            r.peak_memory.rss_mb(),
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");

    if let Some(fastest) = results.iter().max_by(|a, b| {
        a.output_tokens_per_sec()
            .partial_cmp(&b.output_tokens_per_sec())
            .unwrap()
    }) {
        println!(
            "\n  Best decode throughput: {} on '{}' – {:.2} output tokens/sec",
            fastest.backend_name,
            fastest.prompt_name,
            fastest.output_tokens_per_sec(),
        );
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    pub backend_name: String,
    pub model_size: String,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub num_iterations: usize,
    pub total_time: Duration,
    pub avg_time_per_iteration: Duration,
    pub tokens_per_second: f64,
    pub per_iter_latencies_us: Vec<u64>,
    pub p50_latency_us: u64,
    pub p95_latency_us: u64,
    pub estimated_kernels: u64,
    pub estimated_kernel_launch_us: u64,
    pub estimated_bandwidth_gbps: f64,
    pub peak_memory: MemoryStats,
}

impl BenchmarkStats {
    pub fn print_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║          Benchmark Results: {}          ", self.backend_name);
        println!("╠═══════════════════════════════════════════════════════════╣");
        println!("║ Model Size:         {:>40} ║", self.model_size);
        println!("║ Batch Size:         {:>40} ║", self.batch_size);
        println!("║ Sequence Length:    {:>40} ║", self.sequence_length);
        println!("║ Iterations:         {:>40} ║", self.num_iterations);
        println!("╟───────────────────────────────────────────────────────────╢");
        println!(
            "║ Total Time:         {:>37.2?} ║",
            self.total_time
        );
        println!(
            "║ Avg Time/Iter:      {:>37.2?} ║",
            self.avg_time_per_iteration
        );
        if !self.per_iter_latencies_us.is_empty() {
            println!(
                "║ Latency p50/p95:    {:>24} / {:>8} µs ║",
                self.p50_latency_us, self.p95_latency_us
            );
        }
        if self.estimated_kernels > 0 {
            println!(
                "║ Est. kernel launches: {:>29} ║",
                self.estimated_kernels
            );
            println!(
                "║ Est. launch overhead: {:>26} µs ║",
                self.estimated_kernel_launch_us
            );
        }
        if self.estimated_bandwidth_gbps > 0.0 {
            println!(
                "║ Est. bandwidth:     {:>30.2} GB/s ║",
                self.estimated_bandwidth_gbps
            );
        }
        println!(
            "║ Throughput:         {:>34.2} tok/s ║",
            self.tokens_per_second
        );
        println!(
            "║ Peak Memory (RSS):  {:>33.2} MB ║",
            self.peak_memory.rss_mb()
        );
        println!("╚═══════════════════════════════════════════════════════════╝\n");
    }
}

pub fn benchmark_model<B: Backend>(
    model: &Gpt<B>,
    config: &GptConfig,
    batch_size: usize,
    seq_len: usize,
    num_iterations: usize,
    device: &B::Device,
    backend_name: &str,
    model_size: &str,
) -> BenchmarkStats {
    println!(
        "Running benchmark: {} - {} model (batch={}, seq_len={}, iterations={})",
        backend_name, model_size, batch_size, seq_len, num_iterations
    );

    let input_shape = [batch_size, seq_len];
    let numel = batch_size * seq_len;
    
    println!("  Warming up...");
    for _ in 0..3 {
        let input = random_input::<B>(numel, input_shape, device);
        let _ = model.forward(input);
        let _ = B::sync(device);
    }

    println!("  Benchmarking...");
    let start = Instant::now();
    
    let mut per_iter_latencies_us = Vec::with_capacity(num_iterations);
    let ((), peak_memory) = measure_memory_usage(|| {
        for i in 0..num_iterations {
            let iter_start = Instant::now();
            let input = random_input::<B>(numel, input_shape, device);
            let _ = model.forward(input);
            let _ = B::sync(device);
            per_iter_latencies_us.push(iter_start.elapsed().as_micros() as u64);
            
            if (i + 1) % 10 == 0 {
                print!(".");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
        }
    }).unwrap_or_else(|e| {
        eprintln!("Warning: Failed to measure memory usage: {}", e);
        ((), MemoryStats { rss_bytes: 0, vsz_bytes: 0 })
    });
    
    println!(); 
    let total_time = start.elapsed();
    let avg_time = total_time / num_iterations as u32;
    
    let total_tokens = batch_size * seq_len * num_iterations;
    let tokens_per_second = total_tokens as f64 / total_time.as_secs_f64();
    let (p50_latency_us, p95_latency_us) = compute_percentiles_us(&per_iter_latencies_us);
    let (estimated_kernels, estimated_kernel_launch_us) =
        estimate_kernel_launch_overhead(config);
    let estimated_bandwidth_gbps =
        estimate_bandwidth_gbps(config, batch_size, seq_len, avg_time);

    BenchmarkStats {
        backend_name: backend_name.to_string(),
        model_size: model_size.to_string(),
        batch_size,
        sequence_length: seq_len,
        num_iterations,
        total_time,
        avg_time_per_iteration: avg_time,
        tokens_per_second,
        per_iter_latencies_us,
        p50_latency_us,
        p95_latency_us,
        estimated_kernels,
        estimated_kernel_launch_us,
        estimated_bandwidth_gbps,
        peak_memory,
    }
}

fn compute_percentiles_us(values: &[u64]) -> (u64, u64) {
    if values.is_empty() {
        return (0, 0);
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let p50 = sorted[n / 2];
    let p95 = sorted[(n as f64 * 0.95) as usize];
    (p50, p95)
}

fn estimate_kernel_launch_overhead(config: &GptConfig) -> (u64, u64) {
    let per_kernel_us: u64 = std::env::var("LLM_BENCH_KERNEL_LAUNCH_US")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(5);
    let kernels_per_layer: u64 = 12;
    let base_kernels: u64 = 3; 
    let total_kernels = base_kernels + kernels_per_layer * config.num_layers as u64;
    (total_kernels, total_kernels * per_kernel_us)
}

fn estimate_bandwidth_gbps(
    config: &GptConfig,
    batch_size: usize,
    seq_len: usize,
    iter_time: Duration,
) -> f64 {
    if iter_time.is_zero() {
        return 0.0;
    }
    let b = batch_size as u64;
    let s = seq_len as u64;
    let h = config.hidden_size as u64;
    let v = config.vocab_size as u64;

    let bytes_hidden = b * s * h * 4;
    let bytes_per_layer = bytes_hidden * 2; 
    let bytes_logits = b * s * v * 4;
    let total_bytes = bytes_hidden + bytes_per_layer * config.num_layers as u64 + bytes_logits;

    total_bytes as f64 / iter_time.as_secs_f64() / 1e9
}

fn random_input<B: Backend>(
    numel: usize,
    shape: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let mut rng = rand::thread_rng();
    let data: Vec<i32> = (0..numel).map(|_| rng.gen_range(0..100)).collect();
    Tensor::<B, 2, Int>::from_data(TensorData::new(data, shape), device)
}

pub fn compare_results(results: &[BenchmarkStats]) {
    if results.is_empty() {
        return;
    }

    println!("\n╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           BENCHMARK COMPARISON                                ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:20} │ {:10} │ {:12} │ {:10} │ {:12} ║",
        "Backend", "Model", "Avg Time", "p50 µs", "Throughput"
    );
    println!("╟─────────────────────┼────────────┼──────────────┼────────────┼──────────────╢");

    for result in results {
        println!(
            "║ {:20} │ {:10} │ {:10.2?} │ {:10} │ {:10.2} tok/s ║",
            result.backend_name,
            result.model_size,
            result.avg_time_per_iteration,
            result.p50_latency_us,
            result.tokens_per_second
        );
    }

    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");

    if let Some(fastest) = results.iter().max_by(|a, b| {
        a.tokens_per_second
            .partial_cmp(&b.tokens_per_second)
            .unwrap()
    }) {
        println!(
            "\n🏆 Fastest: {} ({:.2} tokens/second)\n",
            fastest.backend_name, fastest.tokens_per_second
        );
    }
}

pub struct BenchmarkConfig {
    pub batch_size: usize,
    pub sequence_length: usize,
    pub num_iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            sequence_length: 32,
            num_iterations: 50,
        }
    }
}

impl BenchmarkConfig {
    pub fn new(batch_size: usize, sequence_length: usize, num_iterations: usize) -> Self {
        Self {
            batch_size,
            sequence_length,
            num_iterations,
        }
    }

    pub fn quick() -> Self {
        Self {
            batch_size: 2,
            sequence_length: 16,
            num_iterations: 20,
        }
    }

    pub fn thorough() -> Self {
        Self {
            batch_size: 8,
            sequence_length: 64,
            num_iterations: 100,
        }
    }
}

use crate::config::RuntimeConfig;

pub fn print_banner() {
    println!("LLM Benchmarking Suite for Rust\n");
    println!(
        "Two benchmark sections:\n\
         1. Burn framework comparison (custom GPT, random weights) - measures\n\
            framework / backend overhead on identical architecture.\n\
         2. Candle pure-Rust LLM (real GGUF weights) - measures end-to-end\n\
            autoregressive generation on real prompts with the same metrics.\n"
    );
}

pub fn print_system_info() {
    println!("\nSystem Information:");
    println!("  - OS: {}", std::env::consts::OS);
    println!("  - Architecture: {}", std::env::consts::ARCH);
    println!("  - Burn Version: 0.20.1");

    #[cfg(target_os = "macos")]
    println!("  - Note: WGPU on macOS uses Metal backend");

    println!();
}

pub fn print_runtime_info(runtime: &RuntimeConfig) {
    println!("Runtime config:");
    println!("  - section3_only: {}", runtime.section3_only);
    println!("  - candle model selection: {}", runtime.candle_model);
    println!(
        "  - HF token: {}",
        if runtime.hf_token.is_some() {
            "provided"
        } else {
            "not provided"
        }
    );
    println!();
}

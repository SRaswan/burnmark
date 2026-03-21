#![recursion_limit = "256"]

mod benchmark;
mod candle_runner;
mod config;
mod display;
mod model;
mod prompts;
mod sections;
mod training;

use benchmark::BenchmarkConfig;
use config::{parse_cli, should_exit_after_list_models, RuntimeConfig};
use display::{print_banner, print_runtime_info, print_system_info};
use sections::{model_configs, run_section1, run_section2, run_section3};

fn main() {
    let cli = parse_cli();

    if should_exit_after_list_models(cli.list_models) {
        return;
    }

    let runtime = RuntimeConfig::from_cli(&cli);

    print_banner();
    print_system_info();
    print_runtime_info(&runtime);

    let model_configs = model_configs();
    let bench_config = BenchmarkConfig::default();

    if !cli.section3_only {
        run_section1(&model_configs, &bench_config);
    }

    run_section2(&cli, &runtime);
    run_section3(&model_configs, &bench_config);
}

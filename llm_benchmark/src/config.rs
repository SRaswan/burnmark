pub struct CliConfig {
    pub model_key: Option<String>,
    pub list_models: bool,
    pub section3_only: bool,
}

pub struct RuntimeConfig {
    pub section3_only: bool,
    pub candle_model: String,
    pub hf_token: Option<String>,
}

impl RuntimeConfig {
    pub fn from_cli(cli: &CliConfig) -> Self {
        let hf_token = std::env::var("LLM_BENCH_HF_TOKEN")
            .or_else(|_| std::env::var("HF_TOKEN"))
            .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
            .ok();

        let candle_model = cli
            .model_key
            .clone()
            .unwrap_or_else(|| "tinyllama".to_string());

        Self {
            section3_only: cli.section3_only,
            candle_model,
            hf_token,
        }
    }
}

pub fn parse_cli() -> CliConfig {
    let args: Vec<String> = std::env::args().collect();

    let mut config = CliConfig {
        model_key: None,
        list_models: false,
        section3_only: std::env::var("LLM_BENCH_SECTION3_ONLY")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false),
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    config.model_key = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --model requires a value (e.g. --model tinyllama)");
                    std::process::exit(1);
                }
            }
            "--list-models" => {
                config.list_models = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    if config.model_key.is_none() {
        config.model_key = std::env::var("LLM_BENCH_CANDLE_MODEL")
            .ok()
            .or_else(|| std::env::var("LLM_BENCH_MODEL").ok());
    }

    config
}

pub fn should_exit_after_list_models(list_models: bool) -> bool {
    #[cfg(feature = "candle")]
    {
        if list_models {
            crate::candle_runner::print_available_models();
            return true;
        }
    }

    #[cfg(not(feature = "candle"))]
    {
        if list_models {
            println!("\n  Model selection requires the 'candle' feature.");
            println!("  Run with: cargo run --release --features candle -- --list-models\n");
            return true;
        }
    }

    false
}

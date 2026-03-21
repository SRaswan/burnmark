#[cfg(feature = "train")]
pub mod inner {
    use std::fmt::Display;
    use std::time::{Duration, Instant};

    use burn::{
        data::{
            dataloader::{batcher::Batcher, DataLoaderBuilder},
            dataset::Dataset,
        },
        nn::loss::CrossEntropyLossConfig,
        optim::AdamConfig,
        record::CompactRecorder,
        tensor::{backend::{AutodiffBackend, Backend}, Int, Tensor, TensorData},
        train::{
            metric::LossMetric,
            ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        },
    };
    use burn_optim::lr_scheduler::constant::ConstantLr;

    use crate::model::{Gpt, GptConfig, GptTranspose};

    
    #[derive(Clone, Debug)]
    pub struct LMItem {
        pub input_ids: Vec<i32>,
        pub target_ids: Vec<i32>,
    }

    pub struct RandomLMDataset {
        items: Vec<LMItem>,
    }

    impl RandomLMDataset {
        pub fn new(size: usize, seq_len: usize, vocab_size: usize) -> Self {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let items = (0..size)
                .map(|_| {
                    let tokens: Vec<i32> = (0..=seq_len)
                        .map(|_| rng.gen_range(0..vocab_size as i32))
                        .collect();
                    LMItem {
                        input_ids: tokens[..seq_len].to_vec(),
                        target_ids: tokens[1..=seq_len].to_vec(),
                    }
                })
                .collect();
            Self { items }
        }
    }

    impl Dataset<LMItem> for RandomLMDataset {
        fn get(&self, index: usize) -> Option<LMItem> {
            self.items.get(index).cloned()
        }

        fn len(&self) -> usize {
            self.items.len()
        }
    }

    #[derive(Clone, Debug)]
    pub struct LMBatch<B: Backend> {
        pub input_ids: Tensor<B, 2, Int>,   
        pub target_ids: Tensor<B, 2, Int>,  
    }

    #[derive(Clone)]
    pub struct LMBatcher<B: Backend> {
        _b: std::marker::PhantomData<B>,
    }

    impl<B: Backend> LMBatcher<B> {
        pub fn new() -> Self {
            Self { _b: std::marker::PhantomData }
        }
    }

    impl<B: Backend> Batcher<B, LMItem, LMBatch<B>> for LMBatcher<B> {
        fn batch(&self, items: Vec<LMItem>, device: &B::Device) -> LMBatch<B> {
            let batch_size = items.len();
            let seq_len = items[0].input_ids.len();

            let input_flat: Vec<i32> = items
                .iter()
                .flat_map(|i| i.input_ids.iter().copied())
                .collect();
            let target_flat: Vec<i32> = items
                .iter()
                .flat_map(|i| i.target_ids.iter().copied())
                .collect();

            let input_ids = Tensor::<B, 1, Int>::from_data(
                TensorData::new(input_flat, [batch_size * seq_len]),
                device,
            )
            .reshape([batch_size, seq_len]);

            let target_ids = Tensor::<B, 1, Int>::from_data(
                TensorData::new(target_flat, [batch_size * seq_len]),
                device,
            )
            .reshape([batch_size, seq_len]);

            LMBatch { input_ids, target_ids }
        }
    }

    
    impl<B: AutodiffBackend> TrainStep for Gpt<B> {
        type Input = LMBatch<B>;
        type Output = ClassificationOutput<B>;

        fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
            let output = lm_forward(self, batch);
            let grads = output.loss.backward();
            TrainOutput::new(self, grads, output)
        }
    }

    impl<B: Backend> InferenceStep for Gpt<B> {
        type Input = LMBatch<B>;
        type Output = ClassificationOutput<B>;

        fn step(&self, batch: Self::Input) -> Self::Output {
            lm_forward(self, batch)
        }
    }

    impl<B: AutodiffBackend> TrainStep for GptTranspose<B> {
        type Input = LMBatch<B>;
        type Output = ClassificationOutput<B>;

        fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
            let output = lm_forward_transpose_train(self, batch);
            let aux_loss = transpose_repro_loss::<B>(output.output.clone());
            let loss = output.loss.clone() + aux_loss * 0.0001;
            let grads = loss.backward();
            TrainOutput::new(self, grads, output)
        }
    }

    impl<B: Backend> InferenceStep for GptTranspose<B> {
        type Input = LMBatch<B>;
        type Output = ClassificationOutput<B>;

        fn step(&self, batch: Self::Input) -> Self::Output {
            lm_forward_transpose(self, batch)
        }
    }

    fn lm_forward<B: Backend>(
        model: &Gpt<B>,
        batch: LMBatch<B>,
    ) -> ClassificationOutput<B> {
        let [batch_size, seq_len] = batch.input_ids.dims();
        let logits = model.forward(batch.input_ids); 
        let vocab = logits.dims()[2];

        let logits_flat = logits.reshape([batch_size * seq_len, vocab]);
        let targets_flat = batch.target_ids.reshape([batch_size * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat.clone(), targets_flat.clone());

        ClassificationOutput::new(loss, logits_flat, targets_flat)
    }

    fn lm_forward_transpose<B: Backend>(
        model: &GptTranspose<B>,
        batch: LMBatch<B>,
    ) -> ClassificationOutput<B> {
        let [batch_size, seq_len] = batch.input_ids.dims();
        let logits = model.forward_infer(batch.input_ids); 
        let vocab = logits.dims()[2];

        let logits_flat = logits.reshape([batch_size * seq_len, vocab]);
        let targets_flat = batch.target_ids.reshape([batch_size * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat.clone(), targets_flat.clone());

        ClassificationOutput::new(loss, logits_flat, targets_flat)
    }

    fn lm_forward_transpose_train<B: AutodiffBackend>(
        model: &GptTranspose<B>,
        batch: LMBatch<B>,
    ) -> ClassificationOutput<B> {
        let [batch_size, seq_len] = batch.input_ids.dims();
        let logits = model.forward_infer(batch.input_ids); 
        let vocab = logits.dims()[2];

        let logits_flat = logits.reshape([batch_size * seq_len, vocab]);
        let targets_flat = batch.target_ids.reshape([batch_size * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat.clone(), targets_flat.clone());

        ClassificationOutput::new(loss, logits_flat, targets_flat)
    }

    fn transpose_repro_loss<B: AutodiffBackend>(logits_flat: Tensor<B, 2>) -> Tensor<B, 1> {
        let t0 = logits_flat.clone().transpose().matmul(logits_flat);
        let t1 = t0.clone().transpose();
        let t2 = t1.clone() + t0.clone();
        (t2.clone() * t2).mean()
    }

    
    pub fn run_training_benchmark<B, M, F>(
        config: &GptConfig,
        device: B::Device,
        backend_name: &str,
        model_label: &str,
        num_epochs: usize,
        batch_size: usize,
        build_model: F,
    ) where
        B: AutodiffBackend,
        M: Display + 'static,
        M: burn::module::AutodiffModule<B>,
        <M as burn::module::AutodiffModule<B>>::InnerModule:
            InferenceStep<Input = LMBatch<B::InnerBackend>, Output = ClassificationOutput<B::InnerBackend>>,
        M: TrainStep<Input = LMBatch<B>, Output = ClassificationOutput<B>>,
        F: FnOnce(&GptConfig, &B::Device) -> M,
    {
        println!(
            "\n[ Training: {} | {} model | {} epochs | batch={} | seq={} ]",
            backend_name,
            model_label,
            num_epochs,
            batch_size,
            config.max_seq_len,
        );

        let seq_len = config.max_seq_len;
        let vocab_size = config.vocab_size;
        
        let train_ds = RandomLMDataset::new(256, seq_len, vocab_size);
        let valid_ds = RandomLMDataset::new(64, seq_len, vocab_size);
        let train_steps = steps_per_epoch(train_ds.len(), batch_size);
        let valid_steps = steps_per_epoch(valid_ds.len(), batch_size);
        let total_steps = (train_steps + valid_steps) * num_epochs;

        let batcher_train = LMBatcher::<B>::new();
        let batcher_valid = LMBatcher::<B::InnerBackend>::new();

        let train_loader = DataLoaderBuilder::new(batcher_train)
            .batch_size(batch_size)
            .shuffle(42)
            .num_workers(1)
            .build(train_ds);

        let valid_loader = DataLoaderBuilder::new(batcher_valid)
            .batch_size(batch_size)
            .num_workers(1)
            .build(valid_ds);

        let model = build_model(config, &device);
        let optimizer_cfg = AdamConfig::new();
        let optimizer = optimizer_cfg.init::<B, M>();
        let lr_scheduler = ConstantLr::new(1e-4_f64);
        let learner = Learner::new(model, optimizer, lr_scheduler);

        let artifact_dir = format!(
            "/tmp/llm_benchmark_train/{}",
            backend_name.replace(' ', "_")
        );

        let training = SupervisedTraining::new(&artifact_dir, train_loader, valid_loader)
            .metric_train_numeric(LossMetric::<burn_ndarray::NdArray>::new())
            .metric_valid_numeric(LossMetric::<burn_ndarray::NdArray>::new())
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(num_epochs)
            .summary()
            .with_training_strategy(burn::train::TrainingStrategy::SingleDevice(device.clone()));

        let start = Instant::now();
        let _trained_model = training.launch(learner);
        let total_time = start.elapsed();
        let avg_step_time = if total_steps > 0 {
            total_time / total_steps as u32
        } else {
            Duration::ZERO
        };

        println!(
            "\n  Training complete. Checkpoints written to: {}\n",
            artifact_dir
        );

        let (estimated_kernels, estimated_kernel_launch_us) =
            estimate_kernel_launch_overhead_train(config);
        let estimated_bandwidth_gbps =
            estimate_bandwidth_gbps_train(config, batch_size, seq_len, avg_step_time);
        print_training_estimates_table(
            estimated_kernels,
            estimated_kernel_launch_us,
            estimated_bandwidth_gbps,
            avg_step_time,
        );
    }

    fn steps_per_epoch(len: usize, batch_size: usize) -> usize {
        if batch_size == 0 {
            return 0;
        }
        (len + batch_size - 1) / batch_size
    }

    fn estimate_kernel_launch_overhead_train(config: &GptConfig) -> (u64, u64) {
        let per_kernel_us: u64 = std::env::var("LLM_BENCH_KERNEL_LAUNCH_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(5);
        let kernels_per_layer: u64 = 12;
        let base_kernels: u64 = 3; 
        let forward_kernels = base_kernels + kernels_per_layer * config.num_layers as u64;
        let train_kernels = forward_kernels * 3; 
        (train_kernels, train_kernels * per_kernel_us)
    }

    fn estimate_bandwidth_gbps_train(
        config: &GptConfig,
        batch_size: usize,
        seq_len: usize,
        avg_step_time: Duration,
    ) -> f64 {
        if avg_step_time.is_zero() {
            return 0.0;
        }
        let b = batch_size as u64;
        let s = seq_len as u64;
        let h = config.hidden_size as u64;
        let v = config.vocab_size as u64;

        let bytes_hidden = b * s * h * 4;
        let bytes_per_layer = bytes_hidden * 2; 
        let bytes_logits = b * s * v * 4;
        let forward_bytes =
            bytes_hidden + bytes_per_layer * config.num_layers as u64 + bytes_logits;
        let train_bytes = forward_bytes * 3; 

        train_bytes as f64 / avg_step_time.as_secs_f64() / 1e9
    }

    fn print_training_estimates_table(
        estimated_kernels: u64,
        estimated_kernel_launch_us: u64,
        estimated_bandwidth_gbps: f64,
        avg_step_time: Duration,
    ) {
        println!("\n  Training Estimate Summary");
        println!("  ┌───────────────────────────────┬──────────────────────────┐");
        println!("  │ Metric                        │ Value                    │");
        println!("  ├───────────────────────────────┼──────────────────────────┤");
        println!(
            "  │ Est. kernel launches/step     │ {:>24} │",
            estimated_kernels
        );
        println!(
            "  │ Est. launch overhead/step     │ {:>21} µs │",
            estimated_kernel_launch_us
        );
        println!(
            "  │ Avg step time                 │ {:>21.2?} │",
            avg_step_time
        );
        let (bw_value, bw_unit) = format_bandwidth(estimated_bandwidth_gbps);
        println!(
            "  │ Est. bandwidth (avg step)     │ {:>19.2} {:<4} │",
            bw_value, bw_unit
        );
        println!("  └───────────────────────────────┴──────────────────────────┘\n");
    }

    fn format_bandwidth(gbps: f64) -> (f64, &'static str) {
        if gbps >= 1.0 {
            (gbps, "GB/s")
        } else {
            (gbps * 1024.0, "MB/s")
        }
    }

}

#[cfg(not(feature = "train"))]
pub mod inner {}

#[cfg(feature = "train")]
pub use inner::run_training_benchmark;

use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig,
        Linear, LinearConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
};

#[derive(Config, Debug)]
pub struct GptConfig {
    #[config(default = 1000)]
    pub vocab_size: usize,
    #[config(default = 256)]
    pub hidden_size: usize,
    
    #[config(default = 4)]
    pub num_layers: usize,
    
    #[config(default = 4)]
    pub num_heads: usize,
    
    #[config(default = 128)]
    pub max_seq_len: usize,
    
    #[config(default = 0.1)]
    pub dropout: f64,
    
    #[config(default = 1024)]
    pub intermediate_size: usize,
}

impl GptConfig {
    pub fn tiny() -> Self {
        Self::new()
            .with_vocab_size(512)
            .with_hidden_size(128)
            .with_num_layers(2)
            .with_num_heads(2)
            .with_max_seq_len(64)
            .with_intermediate_size(256)
    }

    pub fn small() -> Self {
        Self::new()
            .with_vocab_size(2048)
            .with_hidden_size(256)
            .with_num_layers(4)
            .with_num_heads(4)
            .with_max_seq_len(128)
            .with_intermediate_size(512)
    }
}

#[derive(Module, Debug)]
pub struct Gpt<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    layers: Vec<TransformerBlock<B>>,
    ln_final: LayerNorm<B>,
    lm_head: Linear<B>,
    dropout: Dropout,
    max_seq_len: usize,
    hidden_size: usize,
}

impl<B: Backend> Gpt<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
            .init(device);
        
        let position_embedding = EmbeddingConfig::new(config.max_seq_len, config.hidden_size)
            .init(device);
        
        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            layers.push(TransformerBlock::new(config, device));
        }
        
        let ln_final = LayerNormConfig::new(config.hidden_size).init(device);
        
        let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);
        
        let dropout = DropoutConfig::new(config.dropout).init();
        
        Self {
            token_embedding,
            position_embedding,
            layers,
            ln_final,
            lm_head,
            dropout,
            max_seq_len: config.max_seq_len,
            hidden_size: config.hidden_size,
        }
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        assert!(seq_len <= self.max_seq_len, "Sequence length exceeds maximum");
        
        let positions = Tensor::arange(0..seq_len as i64, &input_ids.device())
            .reshape([1, seq_len])
            .repeat(&[batch_size, 1]);
        
        let token_emb = self.token_embedding.forward(input_ids);
        let pos_emb = self.position_embedding.forward(positions);
        
        let mut hidden = token_emb + pos_emb;
        hidden = self.dropout.forward(hidden);
        
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        
        hidden = self.ln_final.forward(hidden);
        self.lm_head.forward(hidden)
    }

    pub fn generate_logits(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let logits = self.forward(input_ids);
        
        let [_batch_size, seq_len, _vocab_size] = logits.dims();
        logits
            .slice([0..1, (seq_len - 1)..seq_len])
            .squeeze_dim::<2>(1)
    }
}

#[derive(Module, Debug)]
pub struct GptTranspose<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    layers: Vec<TransformerBlock<B>>,
    mix: Linear<B>,
    ln_final: LayerNorm<B>,
    dropout: Dropout,
    max_seq_len: usize,
    hidden_size: usize,
    vocab_size: usize,
}

impl<B: Backend> GptTranspose<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
            .init(device);

        let position_embedding = EmbeddingConfig::new(config.max_seq_len, config.hidden_size)
            .init(device);

        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            layers.push(TransformerBlock::new(config, device));
        }

        let mix = LinearConfig::new(config.hidden_size, config.hidden_size).init(device);

        let ln_final = LayerNormConfig::new(config.hidden_size).init(device);
        let dropout = DropoutConfig::new(config.dropout).init();

        Self {
            token_embedding,
            position_embedding,
            layers,
            mix,
            ln_final,
            dropout,
            max_seq_len: config.max_seq_len,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
        }
    }

    fn forward_hidden(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        assert!(seq_len <= self.max_seq_len, "Sequence length exceeds maximum");

        let device = input_ids.device();

        let positions = Tensor::arange(0..seq_len as i64, &device)
            .reshape([1, seq_len])
            .repeat(&[batch_size, 1]);

        let token_emb = self.token_embedding.forward(input_ids);
        let pos_emb = self.position_embedding.forward(positions);

        let mut hidden = token_emb + pos_emb;
        hidden = self.dropout.forward(hidden);

        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }

        self.ln_final.forward(hidden)
    }

    fn logits_from_hidden(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, h] = hidden.dims();
        let hidden_2d = hidden.reshape([b * s, h]);
        let t0 = self.mix.weight.val();
        let t1 = t0.clone().transpose();
        let t2 = t1.clone() + t0.clone();
        let mixed = hidden_2d.matmul(t2);
        let weight_t = self.token_embedding.weight.val().transpose();
        let logits_2d = mixed.matmul(weight_t);

        logits_2d.reshape([b, s, self.vocab_size])
    }

    pub fn forward_infer(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let hidden = self.forward_hidden(input_ids);
        self.logits_from_hidden(hidden)
    }

    pub fn generate_logits(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let logits = self.forward_infer(input_ids);

        let [_batch_size, seq_len, _vocab_size] = logits.dims();
        logits
            .slice([0..1, (seq_len - 1)..seq_len])
            .squeeze_dim::<2>(1)
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: MultiHeadAttention<B>,
    ln1: LayerNorm<B>,
    mlp: Mlp<B>,
    ln2: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        let attention = MultiHeadAttentionConfig::new(config.hidden_size, config.num_heads)
            .with_dropout(config.dropout)
            .init(device);
        
        let ln1 = LayerNormConfig::new(config.hidden_size).init(device);
        let ln2 = LayerNormConfig::new(config.hidden_size).init(device);
        let mlp = Mlp::new(config, device);
        let dropout = DropoutConfig::new(config.dropout).init();
        
        Self {
            attention,
            ln1,
            mlp,
            ln2,
            dropout,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        
        let normed = self.ln1.forward(x.clone());
        let attn_input = MhaInput::self_attn(normed);
        let attn_out = self.attention.forward(attn_input);
        let x = x + self.dropout.forward(attn_out.context);
        
        let normed = self.ln2.forward(x.clone());
        let mlp_out = self.mlp.forward(normed);
        x + self.dropout.forward(mlp_out)
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(config.hidden_size, config.intermediate_size).init(device);
        let fc2 = LinearConfig::new(config.intermediate_size, config.hidden_size).init(device);
        let dropout = DropoutConfig::new(config.dropout).init();
        
        Self { fc1, fc2, dropout }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::gelu(x);
        let x = self.dropout.forward(x);
        self.fc2.forward(x)
    }
}

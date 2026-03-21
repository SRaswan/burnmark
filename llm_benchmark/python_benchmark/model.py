import torch
import torch.nn as nn
import torch.nn.functional as F


class GptConfig:
    def __init__(
        self,
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        max_seq_len=128,
        dropout=0.1,
        intermediate_size=1024,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.intermediate_size = intermediate_size

    @classmethod
    def tiny(cls):
        return cls(
            vocab_size=512,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
            max_seq_len=64,
            intermediate_size=256,
        )

    @classmethod
    def small(cls):
        return cls(
            vocab_size=2048,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            max_seq_len=128,
            intermediate_size=512,
        )


class TransformerBlock(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.dropout(attn_out)

        normed = self.ln2(x)
        mlp_out = self.fc2(self.dropout(F.gelu(self.fc1(normed))))
        x = x + self.dropout(mlp_out)
        return x


class Gpt(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.ln_final = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("Sequence length exceeds maximum")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)

        hidden = token_emb + pos_emb
        hidden = self.dropout(hidden)

        for layer in self.layers:
            hidden = layer(hidden)

        hidden = self.ln_final(hidden)
        return self.lm_head(hidden)

    def generate_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        return logits[:, -1, :]
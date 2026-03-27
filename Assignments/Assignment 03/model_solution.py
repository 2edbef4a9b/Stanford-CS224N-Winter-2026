"""
A bare-bones GPT-2 style transformer.
"""

import math
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformers import GPT2LMHeadModel

from utils import state_dict_converter

# TODO: Add in attention mask to the entire assignment
# TODO: Maybe add KV caching


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    context_length: int
    vocab_size: int


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # Using attention dim from attention is all you need
        assert config.d_model % config.n_heads == 0
        self.d_attention = int(config.d_model / config.n_heads)

        # self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)

        self.W_k = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_q = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_v = nn.Linear(config.d_model, self.d_attention * config.n_heads)

        self.W_o = nn.Linear(self.d_attention * config.n_heads, config.d_model)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
            persistent=False,
        )

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        # Calculate queries, keys, values
        w_k = self.W_k(x)
        w_q = self.W_q(x)
        w_v = self.W_v(x)

        # Get batch size and sequence length
        batch_size, seq_len, _ = x.shape

        # Reshape to (batch, seq_len, n_heads, d_attention)
        w_k = w_k.view(batch_size, seq_len, -1, self.d_attention)
        w_q = w_q.view(batch_size, seq_len, -1, self.d_attention)
        w_v = w_v.view(batch_size, seq_len, -1, self.d_attention)

        # Transpose to (batch, n_heads, seq_len, d_attention)
        w_k = w_k.transpose(1, 2)
        w_q = w_q.transpose(1, 2)
        w_v = w_v.transpose(1, 2)

        # Calculate attention scores
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = torch.matmul(w_q, w_k.transpose(-2, -1)) / math.sqrt(self.d_attention)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Apply softmax to get attention weights
        weights = torch.softmax(scores, dim=-1)
        hidden_states = torch.matmul(weights, w_v)

        # Transpose back to (batch, seq_len, n_heads, d_attention)
        hidden_states = hidden_states.transpose(1, 2).contiguous()

        # Reshape back to (batch, seq_len, d_model)
        hidden_states = hidden_states.view(batch_size, seq_len, -1)

        return self.W_o(hidden_states)


class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Float[Tensor, ...]) -> Float[Tensor, ...]:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))  # fmt: skip


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.gelu = GELU()

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mlp = MLP(config)
        self.attention = CausalAttention(config)
        self.pre_layer_norm = nn.LayerNorm(config.d_model)
        self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        x = x + self.attention(self.pre_layer_norm(x))
        x = x + self.mlp(self.post_layer_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.context_length, config.d_model)
        self.backbone = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                )

    def forward(
        self, x: Int[Tensor, "batch_size seq_len"]
    ) -> Float[Tensor, "batch seq_len vocab_size"]:

        # Token embeddings
        embeddings = self.embeddings(x)

        # Position embeddings
        positions = torch.arange(start=0, end=x.shape[1], device=x.device)
        position_embeddings = self.position_embeddings(positions)
        hidden_states = embeddings + position_embeddings

        # Decoder blocks
        for block in self.backbone:
            hidden_states = block(hidden_states)

        # Final layer norm and language modeling head
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    @torch.no_grad()
    def generate(
        self,
        x: Int[Tensor, "batch_size seq_len"],
        num_new_tokens: int,
    ) -> Int[Tensor, "batch_size seq_len+num_new_tokens"]:

        for _ in range(num_new_tokens):
            logits = self(x)
            last_logit = logits[:, -1, :]
            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)
            x = torch.cat((x, next_token), dim=-1)

        return x

    def get_loss_on_batch(
        self,
        input_ids: Int[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, ""]:

        # Construct inputs and targets for next token prediction
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        # Compute logits
        logits = self(inputs)

        # Flatten logits and targets for loss computation
        targets = targets.flatten(0, 1)
        logits = logits.flatten(0, 1)

        return torch.nn.functional.cross_entropy(logits, targets)

    @classmethod
    def from_pretrained(cls):
        """
        We simply always load up the GPT-2 model
        """

        # Config for GPT-2
        config = ModelConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            context_length=1024,
            vocab_size=50257,
        )

        model = cls(config)

        # Load weights from HuggingFace
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        converted_state_dict: dict[str, Tensor] = state_dict_converter(
            model_hf.state_dict()
        )

        model.load_state_dict(converted_state_dict)

        return model


if __name__ == "__main__":
    # Uncomment this if you are not logged in
    # huggingface_hub.login()

    model = Transformer.from_pretrained()

"""Small LLaMA-like causal LM, matching the local SmolLM2-135M config.

This is a compact training implementation in the style of the local training models.
It intentionally omits generation/cache/quantization features that are not
needed for this pretraining code path.

Source for default hyperparameters:
- ../SmolLM2-135M/config.json
"""

from dataclasses import dataclass, fields
import inspect
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import functional as F


@dataclass
class LlamaConfig:
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    max_position_embeddings: int = 8192
    block_size: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    attention_dropout: float = 0.0
    attention_bias: bool = False
    mlp_bias: bool = False
    qk_norm: bool = False
    initializer_range: float = 1.0 / 24.0
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False
    # "standard": use initializer_range from the SmolLM2 config for all
    # Linear/Embedding weights. "gpt2_scaled": use GPT-2-style std=0.02, then
    # shrink residual-branch output projections by 1/sqrt(2 * num_hidden_layers).
    init_method: str = "standard"

    @classmethod
    def from_dict(cls, data: dict) -> "LlamaConfig":
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    @classmethod
    def from_json(cls, path: str | Path) -> "LlamaConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    batch, kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, kv_heads, repeats, seq_len, head_dim)
    return x.reshape(batch, kv_heads * repeats, seq_len, head_dim)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(y)


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if config.init_method not in {"standard", "gpt2_scaled"}:
            raise ValueError("init_method must be 'standard' or 'gpt2_scaled'")

        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = config.gradient_checkpointing

        self.apply(self._init_weights)
        if config.init_method == "gpt2_scaled":
            self._apply_gpt2_residual_scaling()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        print(f"init method: {config.init_method}")
        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def _base_init_std(self) -> float:
        if self.config.init_method == "standard":
            return self.config.initializer_range
        if self.config.init_method == "gpt2_scaled":
            return 0.02
        raise ValueError(f"unknown init_method: {self.config.init_method}")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self._base_init_std())
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self._base_init_std())

    def _apply_gpt2_residual_scaling(self) -> None:
        residual_std = 0.02 / math.sqrt(2 * self.config.num_hidden_layers)
        for layer in self.layers:
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=residual_std)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(f"sequence length {seq_len} exceeds {self.config.max_position_embeddings}")

        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(x, seq_len)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin)
        x = self.norm(x)

        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, lr, betas, device_type, adam_eps=1e-8):
        params = {name: p for name, p in self.named_parameters() if p.requires_grad}
        decay = [p for _, p in params.items() if p.dim() >= 2]
        no_decay = [p for _, p in params.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay, "weight_decay": weight_decay, "optimizer_name": "adamw"},
            {"params": no_decay, "weight_decay": 0.0, "optimizer_name": "adamw"},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra_args = {"fused": True} if fused_available and device_type == "cuda" else {}
        print(f"num decayed parameter tensors: {len(decay)}, with {sum(p.numel() for p in decay):,} parameters")
        print(f"num non-decayed parameter tensors: {len(no_decay)}, with {sum(p.numel() for p in no_decay):,} parameters")
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=adam_eps, **extra_args)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model FLOP utilization, matching nanoGPT's rough PaLM-style formula."""
        cfg = self.config
        n_params = self.get_num_params()
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        seq_len = cfg.block_size
        flops_per_token = 6 * n_params + 12 * cfg.num_hidden_layers * cfg.num_attention_heads * head_dim * seq_len
        flops_per_fwdbwd = flops_per_token * seq_len
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

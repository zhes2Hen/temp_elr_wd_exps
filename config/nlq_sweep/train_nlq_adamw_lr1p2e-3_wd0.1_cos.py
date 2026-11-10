# Launch from the repository root on two GPUs:
# $ torchrun --standalone --nproc-per-node=2 train.py config/nlq_sweep/train_nlq_adamw_lr1p2e-3_wd0.1_cos.py

# SmolLM2-135M with learnable QK-Norm, trained with cosine LR decay.

import math


out_dir = "../out/nlq_sweep/nlq_adamw_lr1p2e-3_wd0.1_cos"

model_name = "llama"
model_config = {"qk_norm": True}
dataset = "fineweb_edu_new_llama"
optimizer_name = "adamw"
init_method = "gpt2_scaled"

batch_size = 16
block_size = 1024
gradient_accumulation_steps = 4  # Global microbatches per update; 2 per rank on two GPUs.

weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-8
grad_clip = 1.0

eval_interval = 1000
eval_iters = 250  # Keep 4,096,000 validation tokens per evaluation.
simple_training = True
save_spectral_norms = False
save_row_col_l2_norms = False

# 42,000 steps * 16 * 4 * 1,024 = 2,752,512,000 train tokens.
max_iters = 42_000
peak_lr = 1.2e-3
floor_lr = 1.2e-4
warmup_steps = 1000
decay_steps = max_iters - warmup_steps


def learning_rate_schedule(it):
    if it == "meta":
        return {
            "name": "cosine",
            "peak": peak_lr,
            "floor": floor_lr,
            "warmup_steps": warmup_steps,
            "stable_steps": 0,
            "decay_steps": decay_steps,
            "decay_type": "cosine",
        }
    it = int(it)
    if it < warmup_steps:
        return peak_lr * (it + 1) / warmup_steps
    progress = min(1.0, (it - warmup_steps) / max(1, decay_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor_lr + coeff * (peak_lr - floor_lr)

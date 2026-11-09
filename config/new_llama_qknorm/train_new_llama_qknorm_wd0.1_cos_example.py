# Example: SmolLM2-135M with QK-Norm, AdamW, and cosine LR decay.
# From the repository root on a server with two allocated GPUs:
# torchrun --standalone --nproc-per-node=2 train.py config/new_llama_qknorm/train_new_llama_qknorm_wd0.1_cos_example.py
# Set CUDA_VISIBLE_DEVICES before launch to select a pair on a shared node.
# For concurrent experiments, use a different GPU pair and out_dir for each run.
# Place train.bin, val.bin, and meta.pkl in data/fineweb_edu_new_llama/.

import math


out_dir = "out/new_llama_qknorm/new_llama_qknorm_wd0.1_cos_example"

model_name = "llama"
model_config = {"qk_norm": True}
dataset = "fineweb_edu_new_llama"
optimizer_name = "adamw"
init_method = "gpt2_scaled"

batch_size = 4
block_size = 1024
gradient_accumulation_steps = 16  # Global: each of two ranks accumulates 8 microbatches.

weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-8
grad_clip = 1.0

eval_interval = 1000
eval_iters = 1000
save_spectral_norms = False
save_row_col_l2_norms = False

# 42,000 steps * 4 * 16 * 1,024 = 2,752,512,000 train tokens.
max_iters = 42_000
peak_lr = 6e-4
floor_lr = 6e-5
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

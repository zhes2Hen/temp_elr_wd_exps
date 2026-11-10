# Launch from the repository root on two GPUs:
# $ torchrun --standalone --nproc-per-node=2 train.py config/nlq_sweep/train_nlq_adamw_lr1p5e-4_wd0.1_7111.py

# SmolLM2-135M with learnable QK-Norm and a 70%-10%-10%-10% LR schedule.

out_dir = "../out/nlq_sweep/nlq_adamw_lr1p5e-4_wd0.1_7111"

model_name = "llama"
model_config = {"qk_norm": True}
dataset = "fineweb_edu_new_llama"
optimizer_name = "adamw"
init_method = "gpt2_scaled"

batch_size = 4
block_size = 1024
gradient_accumulation_steps = 16  # Global microbatches per update; 8 per rank on two GPUs.

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
peak_lr = 1.5e-4
warmup_steps = 1000
lr_drop_ratio = 10 ** (-1 / 3)
stage_2_start_iter = max_iters * 7 // 10
stage_3_start_iter = max_iters * 8 // 10
stage_4_start_iter = max_iters * 9 // 10


def learning_rate_schedule(it):
    if it == "meta":
        return {
            "name": "7-1-1-1",
            "peak": peak_lr,
            "floor": peak_lr * lr_drop_ratio**3,
            "warmup_steps": warmup_steps,
            "stage_start_iters": [0, stage_2_start_iter, stage_3_start_iter, stage_4_start_iter],
            "stage_lr_ratios": ["1", "10^(-1/3)", "10^(-2/3)", "10^(-1)"],
            "decay_type": "piecewise_constant",
        }
    it = int(it)
    if it < warmup_steps:
        return peak_lr * (it + 1) / warmup_steps
    if it < stage_2_start_iter:
        return peak_lr
    if it < stage_3_start_iter:
        return peak_lr * lr_drop_ratio
    if it < stage_4_start_iter:
        return peak_lr * lr_drop_ratio**2
    return peak_lr * lr_drop_ratio**3

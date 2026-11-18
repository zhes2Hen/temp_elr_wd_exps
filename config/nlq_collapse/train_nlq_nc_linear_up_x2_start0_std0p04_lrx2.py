# Launch from the repository root on two GPUs:
# $ torchrun --standalone --nproc-per-node=2 train.py config/nlq_collapse/train_nlq_nc_linear_up_x2_start0_std0p04_lrx2.py

out_dir = "../out/nlq_collapse/nlq_nc_linear_up_x2_start0_std0p04_lrx2"

model_name = "llama"
model_config = {"qk_norm": True, "gpt2_initializer_range": 0.04}
dataset = "openwebtext_new_llama"
optimizer_name = "norm_control_adam"
init_method = "gpt2_scaled"

batch_size = 32
block_size = 1024
gradient_accumulation_steps = 2  # Global microbatches per update; 1 per rank on two GPUs.

weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-8
grad_clip = 1.0

eval_interval = 1000
eval_iters = 125  # Keep 4,096,000 validation tokens per evaluation.
simple_training = True
save_spectral_norms = False
save_row_col_l2_norms = False

# 42,000 steps * 32 * 2 * 1,024 = 2,752,512,000 train tokens.
max_iters = 42_000
peak_lr = 2 * 6e-4
floor_lr = 2 * 6e-5
warmup_steps = 1000
control_start_iter = 0
decay_steps = max_iters // 10
stable_steps = max_iters - warmup_steps - decay_steps
final_norm_ratio = 2.0


def base_wsd_lr(it):
    if it == "meta":
        return {
            "name": "WSD",
            "peak": peak_lr,
            "floor": floor_lr,
            "warmup_steps": warmup_steps,
            "stable_steps": stable_steps,
            "decay_steps": decay_steps,
            "decay_ratio": 0.10,
            "decay_type": "linear",
        }
    it = int(it)
    if it < warmup_steps:
        return peak_lr * (it + 1) / warmup_steps
    if it < warmup_steps + stable_steps:
        return peak_lr
    progress = min(1.0, (it - warmup_steps - stable_steps) / max(1, decay_steps))
    return floor_lr + (1.0 - progress) * (peak_lr - floor_lr)


def norm_ratio(it):
    """Target norm after iteration it, relative to the initialization norm."""
    if it < control_start_iter:
        return 1.0
    progress = min(1.0, (it - control_start_iter + 1) / (max_iters - control_start_iter))
    return 1.0 + progress * (final_norm_ratio - 1.0)


def learning_rate_schedule(it):
    if it == "meta":
        meta = dict(base_wsd_lr("meta"))
        meta.update({
            "name": "ELR-matched WSD with linear-up norm",
            "control_start_iter": control_start_iter,
            "rule": "matrix_lr(it) = base_wsd_lr(it) * norm_ratio(it - 1)",
            "gamma_lr": "base_wsd_lr",
        })
        return meta
    # Match ELR using the norm entering this update, not its projection target.
    return base_wsd_lr(it) * norm_ratio(it - 1)


gamma_learning_rate_schedule = base_wsd_lr


def norm_control_schedule(it):
    if it == "meta":
        return {
            "enabled": True,
            "name": "linear_up_from_initial_norm",
            "control_start_iter": control_start_iter,
            "reference_iter": control_start_iter - 1,
            "end_ratio": final_norm_ratio,
            "first_controlled_iter_already_moves": True,
        }
    if it < control_start_iter:
        return None
    return norm_ratio(it)

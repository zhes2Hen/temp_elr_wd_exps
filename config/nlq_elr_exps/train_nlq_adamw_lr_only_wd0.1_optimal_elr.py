# Launch from the repository root on two GPUs:
# $ torchrun --standalone --nproc-per-node=2 train.py config/nlq_elr_exps/train_nlq_adamw_lr_only_wd0.1_optimal_elr.py

# SmolLM2-135M with learnable QK-Norm. AdamW with online LR matching to the discrete FSL-optimal ELR.

from pathlib import Path

import numpy as np


out_dir = "../out/nlq_elr_exps/nlq_adamw_lr_only_wd0.1_optimal_elr"

model_name = "llama"
model_config = {"qk_norm": True}
dataset = "fineweb_edu_new_llama"
optimizer_name = "adamw_lr_only"
init_method = "standard"

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
gamma_learning_rate_schedule = None  # Share the dynamic matrix LR.

# The table is already indexed by optimizer step, including warmup.
elr_path = Path(config_file).resolve().with_name("adamh_optimal_elr.npy")
elr_values = np.load(elr_path, allow_pickle=False)
if elr_values.shape != (max_iters,):
    raise ValueError(f"Expected {max_iters} ELR values in {elr_path}, got shape {elr_values.shape}")
if not np.isfinite(elr_values).all() or np.any(elr_values < 0):
    raise ValueError(f"ELR values must be finite and non-negative: {elr_path}")

# The trainer converts this target to a shared LR using the current matrix RMS norms.
target_elr_schedule = make_numpy_schedule(elr_values, elr_path)

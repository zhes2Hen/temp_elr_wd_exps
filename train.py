"""
Decoder-only Llama pretraining on one GPU or with torchrun and DDP.

The main intentional changes from original nanoGPT are:
- use the local Llama model instead of GPT-2 model.py;
- consume train.bin sequentially as fixed-length blocks for one-pass training;
- keep the token budget per optimizer update fixed across GPU counts.
"""

import os
import time
import math
import pickle
import sys
import uuid
from dataclasses import asdict
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model_llama import LlamaConfig, LlamaForCausalLM
from optimizer import (
    attach_post_step_parameter_stats,
    build_optimizer,
    collect_pre_step_parameter_stats,
    optimizer_control_state,
    prepare_optimizer_step,
)
from training_checkpoint import (
    dataset_signature,
    file_digest,
    gather_rng_states,
    optimizer_extra_state,
    read_resume_logs,
    restore_optimizer_extra,
    restore_rng_state,
    trim_step_log,
    validate_checkpoint,
)
from training_metrics import SimpleTrainingMetrics

# -----------------------------------------------------------------------------
# I/O
out_dir = "out/llama_example"
eval_interval = 1000
eval_iters = 1000  # Global number of batches per split, not per rank.
eval_only = False
simple_training = False  # Keep only LR, loss, and aggregate ELR when enabled.
save_checkpoint_only_at_end = False  # Simple metrics follow the checkpoint cadence.
init_from = "scratch"
model_name = "llama"  # Keep this key for existing configs and run metadata.
# Model config can be:
# - None: use the SmolLM2-135M defaults in LlamaConfig.
# - str: path to a JSON config file.
# - dict: already-loaded config data, useful for compact experiment configs.
model_config = None
init_method = "gpt2_scaled"
# data
dataset = "fineweb_edu_new_llama"
batch_size = 4  # micro-batch size in sequences
block_size = 1024
gradient_accumulation_steps = 16  # Global microbatches/update; divided among ranks.
# model/runtime
gradient_checkpointing = False
# optimizer
optimizer_name = "adamw"
max_iters = 42000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-8
grad_clip = 1.0
muon_momentum = 0.95
muon_nesterov = True
muon_update_scale = 0.2
muon_ns_eps = 1e-7
save_spectral_norms = True
save_row_col_l2_norms = True
spectral_norm_power_iters = 8
norm_eps = 1e-12
# system
backend = "nccl"
device = "cuda"
dtype = "bfloat16"
compile = False

def learning_rate_schedule(it):
    """Default WSD schedule. Config files should override this function."""
    schedule = {
        "name": "WSD",
        "peak": 6e-4,
        "floor": 6e-5,
        "warmup_steps": 1000,
        "stable_steps": 0,
        "decay_steps": max_iters - 1000,
        "decay_type": "cosine",
    }
    if it == "meta":
        return schedule
    if it < schedule["warmup_steps"]:
        return schedule["peak"] * (it + 1) / schedule["warmup_steps"]
    decay_progress = min(1.0, (it - schedule["warmup_steps"]) / max(1, schedule["decay_steps"]))
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return schedule["floor"] + cosine * (schedule["peak"] - schedule["floor"])


gamma_learning_rate_schedule = None


def norm_control_schedule(it):
    """Default: no norm control. Config files can override this function."""
    if it == "meta":
        return {"enabled": False}
    return None
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and (isinstance(v, (int, float, bool, str, dict, list)) or v is None)
]
exec(open("configurator.py").read())

if "model_config_path" in globals():
    raise ValueError("model_config_path is obsolete; set model_config to the JSON path instead")
if "matrix_learning_rate_schedule" in globals() or "matrix_norm_control_schedule" in globals():
    raise ValueError("per-group matrix schedules require train_diff.py and are not supported here")
if model_name != "llama":
    raise ValueError(f"this training script only supports model_name='llama', got {model_name!r}")
for key in ("batch_size", "block_size", "max_iters", "eval_interval", "eval_iters", "gradient_accumulation_steps"):
    value = globals()[key]
    if type(value) is not int or value < 1:
        raise ValueError(f"{key} must be a positive integer")
if gamma_learning_rate_schedule is not None and not callable(gamma_learning_rate_schedule):
    raise TypeError("gamma_learning_rate_schedule must be None or a callable schedule function")
if norm_control_schedule is not None and not callable(norm_control_schedule):
    raise TypeError("norm_control_schedule must be None or a callable schedule function")
if not isinstance(simple_training, bool):
    raise TypeError("simple_training must be a bool")
if simple_training and eval_only:
    raise ValueError("simple_training requires training steps; use full mode for eval_only")
if not isinstance(save_checkpoint_only_at_end, bool):
    raise TypeError("save_checkpoint_only_at_end must be a bool")
if not isinstance(save_spectral_norms, bool):
    raise TypeError("save_spectral_norms must be a bool")
if not isinstance(save_row_col_l2_norms, bool):
    raise TypeError("save_row_col_l2_norms must be a bool")
if (
    not isinstance(spectral_norm_power_iters, int)
    or isinstance(spectral_norm_power_iters, bool)
    or spectral_norm_power_iters <= 0
):
    raise ValueError("spectral_norm_power_iters must be a positive integer")
if norm_eps <= 0.0:
    raise ValueError("norm_eps must be positive")
if not 0.0 <= muon_momentum < 1.0:
    raise ValueError("muon_momentum must satisfy 0 <= momentum < 1")
if not isinstance(muon_nesterov, bool):
    raise TypeError("muon_nesterov must be a bool")
if muon_update_scale <= 0.0:
    raise ValueError("muon_update_scale must be positive")
if muon_ns_eps <= 0.0:
    raise ValueError("muon_ns_eps must be positive")
if init_from not in {"scratch", "resume"}:
    raise ValueError("init_from must be 'scratch' or 'resume'")
if globals().get("_resume_requested", False):
    init_from = "resume"
    if eval_only:
        raise ValueError("eval_only is not supported when resuming training")
elif init_from != "scratch":
    raise ValueError("train.py supports scratch training only; use train_resume.py to resume")


def schedule_meta(schedule):
    if schedule is None:
        return None
    return schedule("meta")


config = {k: globals()[k] for k in config_keys if k != "gamma_learning_rate_schedule"}
config["learning_rate_schedule"] = learning_rate_schedule("meta")
config["gamma_learning_rate_schedule"] = schedule_meta(gamma_learning_rate_schedule)
config["norm_control_schedule"] = schedule_meta(norm_control_schedule)
# -----------------------------------------------------------------------------

ddp_world_size = int(os.environ.get("WORLD_SIZE", "1"))
ddp = ddp_world_size > 1
ddp_rank = int(os.environ["RANK"]) if ddp else 0
master_process = ddp_rank == 0
local_accumulation_steps, remainder = divmod(gradient_accumulation_steps, ddp_world_size)
if local_accumulation_steps < 1 or remainder:
    raise ValueError(
        f"gradient_accumulation_steps={gradient_accumulation_steps} must be a positive "
        f"multiple of WORLD_SIZE={ddp_world_size}"
    )
device_type = torch.device(device).type
if dtype != "bfloat16":
    raise ValueError("this script supports dtype='bfloat16' only")
if device_type == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if ddp:
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"BF16 is not supported on {device}")
    if ddp:
        torch.cuda.set_device(device)
if ddp:
    dist.init_process_group(backend=backend)

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + ddp_rank)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# -----------------------------------------------------------------------------
# Data loader: fixed-size, non-overlapping blocks over a flat token stream.
data_dir = os.path.join("data", dataset)
meta_path = os.path.join(data_dir, "meta.pkl")
if not os.path.exists(meta_path):
    raise FileNotFoundError(f"missing dataset metadata: {meta_path}")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)

data_dtype = np.dtype(meta.get("dtype", "uint16"))
meta_vocab_size = meta["vocab_size"]
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=data_dtype, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=data_dtype, mode="r")

batch_tokens = batch_size * block_size
if len(val_data) < batch_tokens + 1:
    raise ValueError(f"val.bin needs at least {batch_tokens + 1:,} tokens for one evaluation batch")
train_stride = ddp_world_size * batch_tokens
tokens_per_iter = gradient_accumulation_steps * batch_tokens
if master_process:
    print(f"dataset dtype: {data_dtype}")
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    print(f"world size: {ddp_world_size}, gradient accumulation per rank: {local_accumulation_steps}")
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

train_pos = 0
ckpt_path = os.path.join(out_dir, "ckpt.pt")
eval_path = os.path.join(out_dir, "eval.pkl")
step_stats_path = os.path.join(out_dir, "step_stats.pkl")
metrics_path = os.path.join(out_dir, "metrics.pkl")
if master_process and init_from == "scratch":
    existing = (ckpt_path, eval_path, step_stats_path, metrics_path) if simple_training else (metrics_path,)
    if any(os.path.lexists(path) for path in existing):
        raise ValueError("out_dir already contains training outputs; use a new directory or train_resume.py")


def make_batch(data, start):
    starts = start + np.arange(batch_size) * block_size
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in starts])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_train_batch():
    global train_pos
    # All ranks advance the same global cursor. Only the offset inside it differs.
    if train_pos + train_stride + 1 > len(train_data):
        raise StopIteration("train.bin exhausted before max_iters; reduce max_iters for one-pass training")
    batch = make_batch(train_data, train_pos + ddp_rank * batch_tokens)
    train_pos += train_stride
    return batch


def get_val_batch(batch_idx):
    max_start = max(1, len(val_data) - batch_tokens - 1)
    start = (batch_idx * batch_tokens) % max_start
    return make_batch(val_data, start)


def load_model_config():
    if model_config is None:
        return LlamaConfig(), "default"
    if isinstance(model_config, str):
        return LlamaConfig.from_json(model_config), model_config
    if isinstance(model_config, dict):
        return LlamaConfig.from_dict(model_config), "dict"
    raise TypeError(f"model_config must be None, str, or dict, got {type(model_config).__name__}")


def build_model_from_config() -> tuple[torch.nn.Module, dict, str]:
    if init_method not in {"standard", "gpt2_scaled"}:
        raise ValueError("currently only init_method='standard' or 'gpt2_scaled' is implemented")

    print("Initializing Llama model from scratch")
    cfg, cfg_source = load_model_config()
    cfg.block_size = block_size
    cfg.gradient_checkpointing = gradient_checkpointing
    cfg.init_method = init_method
    assert cfg.vocab_size == meta_vocab_size, (cfg.vocab_size, meta_vocab_size)
    built_model = LlamaForCausalLM(cfg)
    built_args = {"model_name": model_name, "config": asdict(cfg)}
    return built_model, built_args, cfg_source


# -----------------------------------------------------------------------------
# Model init
iter_num = 0
model, model_args, model_config_source = build_model_from_config()
checkpoint = None
if init_from == "resume":
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"no completed checkpoint to resume: {ckpt_path}")
    # Checkpoints contain Python and NumPy RNG state. Load only trusted files.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

run_meta = {
    "format_version": 1,
    "run_id": (checkpoint["meta"].get("run_id") if checkpoint is not None else None) or uuid.uuid4().hex,
    "model_name": model_name,
    "dataset": dataset,
    "out_dir": out_dir,
    "data_dir": data_dir,
    "model_config_source": model_config_source,
    "tokens_per_iter": tokens_per_iter,
    "batch_size": batch_size,
    "block_size": block_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "world_size": ddp_world_size,
    "device_type": device_type,
    "local_gradient_accumulation_steps": local_accumulation_steps,
    "eval_tokens_per_split": eval_iters * batch_tokens,
    "max_iters": max_iters,
    "data_meta": meta,
    "config": config,
    "config_file_digests": [file_digest(arg) for arg in sys.argv[1:] if "=" not in arg],
    "config_overrides": {
        key: globals()[key]
        for key in (arg[2:].split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--") and "=" in arg)
        if key not in {"init_from", "out_dir", "device", "simple_training"}
    },
    "model_args": model_args,
}
if checkpoint is not None:
    allow_world_size_change = globals().get("_allow_world_size_change", False)
    validate_checkpoint(checkpoint, config, run_meta, model_args, allow_world_size_change=allow_world_size_change)
    iter_num = checkpoint["iter_num"] + 1
    train_pos = checkpoint["train_pos"]
    model.load_state_dict(checkpoint["model"], strict=True)
    previous_world_size = checkpoint["meta"].get("world_size", 1)
    world_size_changed = previous_world_size != ddp_world_size and iter_num < max_iters
    world_size_changes = list(checkpoint["meta"].get("world_size_changes", []))
    if world_size_changed:
        world_size_changes.append({
            "start_iter": iter_num,
            "from_world_size": previous_world_size,
            "to_world_size": ddp_world_size,
            "rng_policy": "reseed_per_rank",
        })
    if world_size_changes:
        run_meta["world_size_changes"] = world_size_changes

if block_size > model.config.max_position_embeddings:
    raise ValueError(f"block_size={block_size} exceeds {model_name} context {model.config.max_position_embeddings}")
requested_tokens = (max_iters - iter_num) * tokens_per_iter
available_tokens = len(train_data) - 1 - train_pos
if requested_tokens > available_tokens:
    raise ValueError(
        f"training request exceeds one-pass data budget: "
        f"{requested_tokens:,} requested tokens > {available_tokens:,} remaining tokens"
    )
if master_process:
    print("Checking the token files used by this run...")
    train_prefix_tokens = min(len(train_data), max(max_iters * tokens_per_iter, eval_iters * batch_tokens) + 1)
    run_meta["data_signature"] = dataset_signature(
        data_dir, len(train_data), len(val_data), train_prefix_tokens, data_dtype.itemsize,
    )
    if checkpoint is not None:
        saved_signature = checkpoint["meta"].get("data_signature")
        if saved_signature is None:
            print("Older checkpoint has no data fingerprint. Make sure both token files are unchanged.")
        elif saved_signature != run_meta["data_signature"]:
            raise ValueError("train.bin or val.bin differs from the checkpoint")

raw_model = model.to(device)
ddp_model = None
if ddp:
    device_ids = [torch.device(device).index] if device_type == "cuda" else None
    ddp_model = DDP(raw_model, device_ids=device_ids)
    model = ddp_model

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = build_optimizer(
    raw_model,
    optimizer_name,
    weight_decay,
    learning_rate_schedule(0),
    (beta1, beta2),
    device_type,
    adam_eps=adam_eps,
    norm_control_schedule=norm_control_schedule,
    muon_momentum=muon_momentum,
    muon_nesterov=muon_nesterov,
    muon_update_scale=muon_update_scale,
    muon_ns_eps=muon_ns_eps,
)
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_optimizer_extra(optimizer, raw_model, optimizer_name, checkpoint)
    scaler.load_state_dict(checkpoint["scaler"])
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)


def atomic_pickle_dump(obj, path):
    tmp_path = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def atomic_torch_save(obj, path):
    tmp_path = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.tmp")
    with open(tmp_path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def iter_unique_parameters(module):
    """Yield each trainable Parameter once, using its first module name."""
    seen_param_ids = set()
    for module_name, submodule in module.named_modules():
        for local_name, param in submodule._parameters.items():
            if param is None or not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen_param_ids:
                continue
            seen_param_ids.add(param_id)
            name = f"{module_name}.{local_name}" if module_name else local_name
            yield {"name": name, "parameter": param}


if master_process:
    parameter_infos = list(iter_unique_parameters(raw_model))
    if simple_training:
        metrics = SimpleTrainingMetrics(max_iters, parameter_infos)
    else:
        optimizer_group_by_param_id = {
            id(param): group
            for group in optimizer.param_groups
            for param in group["params"]
        }


def gamma_lr_for_iter(iter_index, matrix_lr):
    if gamma_learning_rate_schedule is None:
        return matrix_lr
    return float(gamma_learning_rate_schedule(iter_index))


def set_optimizer_learning_rates(optimizer, matrix_lr, vector_lr):
    for param_group in optimizer.param_groups:
        parameter_kind = param_group.get("parameter_kind", "matrix")
        if parameter_kind == "vector":
            param_group["lr"] = vector_lr
        else:
            param_group["lr"] = matrix_lr


eval_history = []


def initialize_step_stats_file():
    header = {
        "meta": run_meta,
        "format": "pickle_stream",
        "record_type": "step_stats",
    }
    with open(step_stats_path, "wb") as f:
        pickle.dump(header, f)


def append_step_record(record):
    with open(step_stats_path, "ab") as f:
        pickle.dump(record, f)
        f.flush()
        os.fsync(f.fileno())


def save_eval_record(record):
    eval_history.append(record)
    atomic_pickle_dump({"meta": run_meta, "eval": eval_history}, eval_path)


def save_checkpoint(eval_record, rng_by_rank):
    checkpoint = {
        "checkpoint_version": 2,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": eval_record["iter"],
        "train_pos": train_pos,
        "config": config,
        "meta": run_meta,
        "eval": eval_record,
        "optimizer_resume": optimizer_extra_state(optimizer, raw_model, optimizer_name),
        "scaler": scaler.state_dict(),
        "rng_by_rank": rng_by_rank,
        **rng_by_rank[0],  # Preserve the old top-level RNG fields for checkpoint readers.
    }
    atomic_torch_save(checkpoint, ckpt_path)


@torch.no_grad()
def estimate_loss():
    out = {}
    raw_model.eval()
    splits = (("val", "eval_val"),) if simple_training else (("train", "eval_train"), ("val", "eval_val"))
    for split, out_key in splits:
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        for k in range(ddp_rank, eval_iters, ddp_world_size):
            if split == "train":
                start = (k * batch_tokens) % max(1, len(train_data) - batch_tokens - 1)
                X, Y = make_batch(train_data, start)
            else:
                X, Y = get_val_batch(k)
            with ctx:
                # Some ranks can have one fewer batch. The raw model avoids
                # DDP forward collectives when batch counts differ.
                _, loss = raw_model(X, Y)
            loss_sum += loss.detach()
        if ddp:
            dist.reduce(loss_sum, dst=0)
        if master_process:
            out[out_key] = loss_sum.item() / eval_iters
    raw_model.train()
    return out


def evaluate_and_save(eval_iter):
    losses = estimate_loss()
    should_save = not save_checkpoint_only_at_end or eval_iter == max_iters - 1
    rng_by_rank = gather_rng_states(device, ddp_rank, ddp_world_size) if should_save else None
    if not master_process:
        return
    record = {"iter": int(eval_iter), **losses}
    if simple_training:
        print(f"eval {eval_iter}: eval_val {record['eval_val']:.4f}")
        metrics.record_validation(eval_iter, record["eval_val"])
        if should_save:
            # Resume can drop metrics beyond a checkpoint, but not recover missing rows.
            atomic_pickle_dump(metrics.frame(run_meta), metrics_path)
    else:
        print(f"eval {eval_iter}: eval_train {record['eval_train']:.4f}, eval_val {record['eval_val']:.4f}")
        save_eval_record(record)
    if should_save:
        print(f"saving checkpoint to {out_dir}")
        save_checkpoint(record, rng_by_rank)
    return record


running_mfu = -1.0
if checkpoint is not None:
    restore_rng_state(checkpoint, device, ddp_rank, ddp_world_size, allow_world_size_change=allow_world_size_change)
    if ddp:
        dist.barrier()  # Do not touch the logs until every rank has restored successfully.
    if master_process:
        if simple_training:
            if metrics.restore(metrics_path, checkpoint):
                atomic_pickle_dump(metrics.frame(run_meta), metrics_path)
        else:
            step_end, eval_history, discarded_evals, running_mfu = read_resume_logs(step_stats_path, eval_path, checkpoint)
            trim_step_log(step_stats_path, step_end)
            if discarded_evals:
                atomic_pickle_dump({"meta": run_meta, "eval": eval_history}, eval_path)
        if iter_num == max_iters:
            print(f"Training already completed all {max_iters:,} updates.")
        else:
            if world_size_changed:
                print(f"World size changed from {previous_world_size} to {ddp_world_size}; reseeded each rank.")
            print(f"Resuming after iter {iter_num - 1}: next iter {iter_num}, train_pos={train_pos:,}")
    del checkpoint
elif master_process and not simple_training:
    initialize_step_stats_file()
if not simple_training:
    t0 = time.time()
local_iter_num = iter_num
while iter_num < max_iters:
    lr = learning_rate_schedule(iter_num)
    gamma_lr = gamma_lr_for_iter(iter_num, lr)
    set_optimizer_learning_rates(optimizer, lr, gamma_lr)
    prepare_optimizer_step(optimizer, iter_num)

    if eval_only:
        evaluate_and_save(iter_num)
        break

    step_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    for micro_step in range(local_accumulation_steps):
        X, Y = get_train_batch()
        # DDP must see both the forward and backward inside no_sync().
        sync_context = ddp_model.no_sync() if ddp and micro_step < local_accumulation_steps - 1 else nullcontext()
        with sync_context:
            with ctx:
                _, loss = model(X, Y)
            step_loss_sum += loss.detach().float()
            scaler.scale(loss / local_accumulation_steps).backward()
    if ddp:
        dist.reduce(step_loss_sum, dst=0)

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else:
        total_grad_norm = None

    if master_process and not simple_training:
        parameter_stats = collect_pre_step_parameter_stats(
            parameter_infos,
            optimizer,
            optimizer_group_by_param_id,
            optimizer_name,
            save_spectral_norms=save_spectral_norms,
            save_row_col_l2_norms=save_row_col_l2_norms,
            spectral_norm_power_iters=spectral_norm_power_iters,
            norm_eps=norm_eps,
        )
    scaler.step(optimizer)
    scaler.update()
    if master_process and not simple_training:
        attach_post_step_parameter_stats(
            parameter_stats,
            parameter_infos,
            save_spectral_norms=save_spectral_norms,
            save_row_col_l2_norms=save_row_col_l2_norms,
            spectral_norm_power_iters=spectral_norm_power_iters,
            norm_eps=norm_eps,
        )
    optimizer.zero_grad(set_to_none=True)

    if simple_training:
        if master_process:
            metrics.record_step(iter_num, lr, step_loss_sum, gradient_accumulation_steps)
    else:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if master_process:
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * local_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            train_loss = step_loss_sum.item() / gradient_accumulation_steps
            step_record = {
                "iter": int(iter_num),
                "train_loss": float(train_loss),
                "lr": float(lr),
                "gamma_lr": float(gamma_lr),
                "tokens": int(train_pos),
                "time_sec": float(dt),
                "mfu": float(running_mfu),
                "global_grad_norm": None if total_grad_norm is None else total_grad_norm.item(),
                "optimizer_control": optimizer_control_state(optimizer),
                "parameters": parameter_stats,
            }
            append_step_record(step_record)
            print(
                f"iter {iter_num}: train loss {train_loss:.4f}, lr {lr:.2e}, "
                f"gamma lr {gamma_lr:.2e}, time {dt * 1000:.2f}ms, "
                f"mfu {running_mfu * 100:.2f}%, tokens {train_pos:,}"
            )
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        evaluate_and_save(iter_num)
    iter_num += 1
    local_iter_num += 1

if ddp:
    dist.destroy_process_group()

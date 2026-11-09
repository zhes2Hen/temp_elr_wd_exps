"""
Single-GPU decoder-only pretraining script, kept close to nanoGPT/train.py.

The main intentional changes from original nanoGPT are:
- use local LLaMA-like or Qwen3-style models instead of GPT-2 model.py;
- consume train.bin sequentially as fixed-length blocks for one-pass training;
- keep the old DDP code as comments, because this project currently targets one GPU.
"""

import os
import time
import math
import pickle
from dataclasses import asdict
from contextlib import nullcontext

import numpy as np
import torch

# DDP is intentionally disabled for now. Keep these imports and the init block
# below as reference when multi-GPU training becomes necessary again.
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group

from model_llama import LlamaConfig, LlamaForCausalLM
from model_qwen3 import Qwen3Config, Qwen3ForCausalLM
from optimizer import (
    attach_post_step_parameter_stats,
    build_optimizer,
    collect_pre_step_parameter_stats,
    optimizer_control_state,
    prepare_optimizer_step,
)

# -----------------------------------------------------------------------------
# I/O
out_dir = "out/llama_example"
eval_interval = 1000
eval_iters = 1000
eval_only = False
init_from = "scratch"  # "scratch" or "resume"
model_name = "llama"
# Model config can be:
# - None: use the dataclass defaults in model_llama.py or model_qwen3.py.
# - str: path to a JSON config file.
# - dict: already-loaded config data, useful for compact experiment configs.
model_config = None
init_method = "gpt2_scaled"
# data
dataset = "fineweb_edu_new_llama"
batch_size = 4  # micro-batch size in sequences
block_size = 1024
gradient_accumulation_steps = 16  # 16 * 4 * 1024 = 65,536 tokens/update
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
save_spectral_norms = True
save_row_col_l2_norms = True
spectral_norm_power_iters = 8
norm_eps = 1e-12
# DDP settings kept for future use, inactive in this single-GPU script
backend = "nccl"
# system
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = False  # torch.compile needs Triton/Inductor, which is not reliable in this Windows setup.

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

if gamma_learning_rate_schedule is not None and not callable(gamma_learning_rate_schedule):
    raise TypeError("gamma_learning_rate_schedule must be None or a callable schedule function")
if norm_control_schedule is not None and not callable(norm_control_schedule):
    raise TypeError("norm_control_schedule must be None or a callable schedule function")
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
if init_from not in {"scratch", "resume"}:
    raise ValueError("init_from must be 'scratch' or 'resume'")
if init_from == "resume" and optimizer_name not in {"adamw", "norm_control_adam"}:
    raise ValueError(
        "resume currently supports only optimizer_name='adamw' or "
        "'norm_control_adam'"
    )


def schedule_meta(schedule):
    if schedule is None:
        return None
    return schedule("meta")


config = {k: globals()[k] for k in config_keys if k != "gamma_learning_rate_schedule"}
config["learning_rate_schedule"] = learning_rate_schedule("meta")
config["gamma_learning_rate_schedule"] = schedule_meta(gamma_learning_rate_schedule)
config["norm_control_schedule"] = schedule_meta(norm_control_schedule)
# -----------------------------------------------------------------------------

# Single-GPU path. The original DDP path is intentionally commented out.
ddp = False
master_process = True
seed_offset = 0
ddp_world_size = 1

# ddp = int(os.environ.get("RANK", -1)) != -1
# if ddp:
#     init_process_group(backend=backend)
#     ddp_rank = int(os.environ["RANK"])
#     ddp_local_rank = int(os.environ["LOCAL_RANK"])
#     ddp_world_size = int(os.environ["WORLD_SIZE"])
#     device = f"cuda:{ddp_local_rank}"
#     torch.cuda.set_device(device)
#     master_process = ddp_rank == 0
#     seed_offset = ddp_rank
#     assert gradient_accumulation_steps % ddp_world_size == 0
#     gradient_accumulation_steps //= ddp_world_size

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
assert dtype=='bfloat16' # we only use bfloat16
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_tokens
print(f"dataset dtype: {data_dtype}")
print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
print(f"tokens per iteration will be: {tokens_per_iter:,}")

train_pos = 0
ckpt_path = os.path.join(out_dir, "ckpt.pt")
eval_path = os.path.join(out_dir, "eval.pkl")
step_stats_path = os.path.join(out_dir, "step_stats.pkl")


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
    if train_pos + batch_tokens + 1 > len(train_data):
        raise StopIteration("train.bin exhausted before max_iters; reduce max_iters for one-pass training")
    batch = make_batch(train_data, train_pos)
    train_pos += batch_tokens
    return batch


def get_val_batch(batch_idx):
    max_start = max(1, len(val_data) - batch_tokens - 1)
    start = (batch_idx * batch_tokens) % max_start
    return make_batch(val_data, start)


def load_model_config(config_cls):
    if model_config is None:
        return config_cls(), "default"
    if isinstance(model_config, str):
        return config_cls.from_json(model_config), model_config
    if isinstance(model_config, dict):
        return config_cls.from_dict(model_config), "dict"
    raise TypeError(f"model_config must be None, str, or dict, got {type(model_config).__name__}")


MODEL_REGISTRY = {
    "llama": {
        "config_cls": LlamaConfig,
        "model_cls": LlamaForCausalLM,
        "description": "LLaMA-like SmolLM2-135M architecture",
    },
    "qwen3": {
        "config_cls": Qwen3Config,
        "model_cls": Qwen3ForCausalLM,
        "description": "Qwen3-style architecture",
    },
}


def build_model_from_config() -> tuple[torch.nn.Module, dict, str]:
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(repr(name) for name in MODEL_REGISTRY)
        raise ValueError(f"currently only model_name in {{{supported}}} is implemented")
    if init_method not in {"standard", "gpt2_scaled"}:
        raise ValueError("currently only init_method='standard' or 'gpt2_scaled' is implemented")

    spec = MODEL_REGISTRY[model_name]
    print(f"Initializing {spec['description']} from scratch")
    cfg, cfg_source = load_model_config(spec["config_cls"])
    cfg.block_size = block_size
    cfg.gradient_checkpointing = gradient_checkpointing
    cfg.init_method = init_method
    assert cfg.vocab_size == meta_vocab_size, (cfg.vocab_size, meta_vocab_size)
    built_model = spec["model_cls"](cfg)
    built_args = {"model_name": model_name, "config": asdict(cfg)}
    return built_model, built_args, cfg_source


RESUME_CONFIG_KEYS = (
    "out_dir",
    "model_name",
    "model_config",
    "init_method",
    "dataset",
    "batch_size",
    "block_size",
    "gradient_accumulation_steps",
    "gradient_checkpointing",
    "optimizer_name",
    "max_iters",
    "weight_decay",
    "beta1",
    "beta2",
    "adam_eps",
    "grad_clip",
    "eval_interval",
    "eval_iters",
    "save_spectral_norms",
    "save_row_col_l2_norms",
    "spectral_norm_power_iters",
    "norm_eps",
    "dtype",
    "compile",
    "learning_rate_schedule",
    "gamma_learning_rate_schedule",
    "norm_control_schedule",
)


def validate_resume_checkpoint(checkpoint):
    required = {
        "model",
        "optimizer",
        "model_args",
        "iter_num",
        "train_pos",
        "config",
        "meta",
        "eval",
        "scaler",
        "rng_state",
        "cuda_rng_state",
        "numpy_rng_state",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing resume fields: {sorted(missing)}")

    checkpoint_config = checkpoint["config"]
    checkpoint_optimizer = checkpoint_config.get("optimizer_name")
    if checkpoint_optimizer not in {"adamw", "norm_control_adam"}:
        raise ValueError(
            "checkpoint resume currently supports only AdamW or norm-control Adam, "
            f"got {checkpoint_optimizer!r}"
        )

    mismatches = []
    for key in RESUME_CONFIG_KEYS:
        checkpoint_value = checkpoint_config.get(key, "<missing>")
        current_value = config.get(key, "<missing>")
        if checkpoint_value != current_value:
            mismatches.append((key, checkpoint_value, current_value))
    if mismatches:
        details = "\n".join(
            f"  {key}: checkpoint={old!r}, current={new!r}"
            for key, old, new in mismatches
        )
        raise ValueError(f"resume config does not match checkpoint:\n{details}")

    checkpoint_model_args = checkpoint["model_args"]
    if checkpoint_model_args.get("model_name") != model_name:
        raise ValueError(
            f"checkpoint model_name={checkpoint_model_args.get('model_name')!r} "
            f"does not match current model_name={model_name!r}"
        )
    spec = MODEL_REGISTRY[model_name]
    current_cfg, _ = load_model_config(spec["config_cls"])
    current_cfg.block_size = block_size
    current_cfg.gradient_checkpointing = gradient_checkpointing
    current_cfg.init_method = init_method
    current_model_args = {"model_name": model_name, "config": asdict(current_cfg)}
    if checkpoint_model_args != current_model_args:
        raise ValueError("current model configuration does not match checkpoint model_args")

    checkpoint_meta = checkpoint["meta"]
    expected_meta = {
        "model_name": model_name,
        "dataset": dataset,
        "data_dir": data_dir,
        "tokens_per_iter": tokens_per_iter,
        "batch_size": batch_size,
        "block_size": block_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_iters": max_iters,
    }
    for key, expected_value in expected_meta.items():
        if checkpoint_meta.get(key) != expected_value:
            raise ValueError(
                f"checkpoint metadata mismatch for {key}: "
                f"{checkpoint_meta.get(key)!r} != {expected_value!r}"
            )
    if checkpoint_meta.get("data_meta") != meta:
        raise ValueError("current dataset meta.pkl does not match checkpoint metadata")
    expected_train_tokens = int(meta.get("train_tokens", len(train_data)))
    expected_val_tokens = int(meta.get("val_tokens", len(val_data)))
    if len(train_data) != expected_train_tokens:
        raise ValueError(
            f"train.bin contains {len(train_data):,} tokens, "
            f"but meta.pkl declares {expected_train_tokens:,}"
        )
    if len(val_data) != expected_val_tokens:
        raise ValueError(
            f"val.bin contains {len(val_data):,} tokens, "
            f"but meta.pkl declares {expected_val_tokens:,}"
        )

    completed_iter = int(checkpoint["iter_num"])
    if completed_iter < 0 or completed_iter >= max_iters:
        raise ValueError(
            f"checkpoint iter_num={completed_iter} is outside [0, {max_iters - 1}]"
        )
    eval_record = checkpoint["eval"]
    if int(eval_record.get("iter", -1)) != completed_iter:
        raise ValueError("checkpoint eval record does not match checkpoint iter_num")
    expected_train_pos = (completed_iter + 1) * tokens_per_iter
    if int(checkpoint["train_pos"]) != expected_train_pos:
        raise ValueError(
            f"checkpoint train_pos={checkpoint['train_pos']:,} does not match "
            f"{completed_iter + 1} completed updates ({expected_train_pos:,})"
        )


def build_model_from_checkpoint(checkpoint) -> tuple[torch.nn.Module, dict, str]:
    checkpoint_model_args = checkpoint["model_args"]
    checkpoint_model_name = checkpoint_model_args["model_name"]
    spec = MODEL_REGISTRY[checkpoint_model_name]
    cfg = spec["config_cls"].from_dict(checkpoint_model_args["config"])
    print(
        f"Resuming {spec['description']} from {ckpt_path} "
        f"after iter {int(checkpoint['iter_num'])}"
    )
    built_model = spec["model_cls"](cfg)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    built_model.load_state_dict(state_dict, strict=True)
    cfg_source = checkpoint["meta"].get("model_config_source", "checkpoint")
    return built_model, checkpoint_model_args, cfg_source


# -----------------------------------------------------------------------------
# Model init
iter_num = 0
checkpoint = None
completed_checkpoint_iter = None

if init_from == "scratch":
    model, model_args, model_config_source = build_model_from_config()
elif init_from == "resume":
    if eval_only:
        raise ValueError("eval_only is not supported together with init_from='resume'")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"missing resume checkpoint: {ckpt_path}")
    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    validate_resume_checkpoint(checkpoint)
    model, model_args, model_config_source = build_model_from_checkpoint(checkpoint)
    completed_checkpoint_iter = int(checkpoint["iter_num"])
    iter_num = completed_checkpoint_iter + 1
    train_pos = int(checkpoint["train_pos"])
else:
    raise AssertionError(f"unreachable init_from={init_from!r}")

if block_size > model.config.max_position_embeddings:
    raise ValueError(f"block_size={block_size} exceeds {model_name} context {model.config.max_position_embeddings}")
requested_tokens = (max_iters - iter_num) * tokens_per_iter
available_tokens = len(train_data) - 1 - train_pos
if requested_tokens > available_tokens:
    raise ValueError(
        f"training request exceeds one-pass data budget: "
        f"{requested_tokens:,} requested tokens > {available_tokens:,} remaining tokens"
    )
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = build_optimizer(
    model,
    optimizer_name,
    weight_decay,
    learning_rate_schedule(iter_num),
    (beta1, beta2),
    device_type,
    adam_eps=adam_eps,
    norm_control_schedule=norm_control_schedule,
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)
else:
    unoptimized_model = model

# if ddp:
#     model = DDP(model, device_ids=[ddp_local_rank])

raw_model = unoptimized_model.module if ddp else unoptimized_model


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


def atomic_copy_prefix(path, byte_count):
    tmp_path = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.resume.tmp")
    remaining = int(byte_count)
    with open(path, "rb") as source, open(tmp_path, "wb") as destination:
        while remaining:
            chunk = source.read(min(16 * 1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"unexpected EOF while copying {path}")
            destination.write(chunk)
            remaining -= len(chunk)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(tmp_path, path)


def validate_log_meta(log_meta, checkpoint_meta, source_name):
    for key in ("model_name", "dataset", "out_dir", "tokens_per_iter"):
        if log_meta.get(key) != checkpoint_meta.get(key):
            raise ValueError(
                f"{source_name} metadata mismatch for {key}: "
                f"{log_meta.get(key)!r} != {checkpoint_meta.get(key)!r}"
            )


def pre_step_norm_from_record(parameter_record, target_ratio):
    for projection_key in ("gradient_proj", "update_proj"):
        projection = parameter_record.get(projection_key) or {}
        coefficient = float(projection.get("coefficient", 0.0))
        parallel_fro = float(projection.get("parallel_fro", 0.0))
        if coefficient != 0.0 and math.isfinite(coefficient) and math.isfinite(parallel_fro):
            # _capture_reference_norms() stores a float32 norm via .item().
            # Round the algebraically recovered value back to that exact width.
            return float(np.float32(parallel_fro / abs(coefficient)))

    parameter_stats = parameter_record.get("parameter") or {}
    post_step_norm = parameter_stats.get("fro")
    if post_step_norm is None or target_ratio <= 0.0:
        raise ValueError(
            f"cannot recover reference norm for {parameter_record.get('name')!r}"
        )
    return float(post_step_norm) / target_ratio


def reference_norms_from_controlled_record(record):
    control = record.get("optimizer_control") or {}
    if not control.get("active", False):
        return None, None
    target_ratio = float(control["target_ratio"])
    reference_norms = {}
    for parameter_record in record.get("parameters", []):
        if parameter_record.get("parameter_kind") != "matrix":
            continue
        name = parameter_record["name"]
        reference_norms[name] = pre_step_norm_from_record(parameter_record, target_ratio)
    if not reference_norms:
        raise ValueError("controlled step record contains no matrix reference norms")
    return reference_norms, control.get("reference_iter", int(record["iter"]) - 1)


def reconcile_step_stats_for_resume(checkpoint_iter, checkpoint_train_pos, checkpoint_meta):
    if not os.path.exists(step_stats_path):
        raise FileNotFoundError(f"missing resume step statistics: {step_stats_path}")

    file_size = os.path.getsize(step_stats_path)
    checkpoint_record = None
    checkpoint_end_offset = None
    recovered_reference_norms = None
    recovered_reference_iter = None
    expected_iter = 0

    with open(step_stats_path, "rb") as f:
        try:
            header = pickle.load(f)
        except Exception as exc:
            raise ValueError(f"cannot read step statistics header: {step_stats_path}") from exc
        if header.get("format") != "pickle_stream" or header.get("record_type") != "step_stats":
            raise ValueError(f"unexpected step statistics format: {step_stats_path}")
        validate_log_meta(header.get("meta", {}), checkpoint_meta, "step_stats")

        while f.tell() < file_size:
            record_start = f.tell()
            try:
                record = pickle.load(f)
            except Exception as exc:
                if checkpoint_record is None:
                    raise ValueError(
                        f"step statistics are corrupted before checkpoint iter {checkpoint_iter}"
                    ) from exc
                print(
                    f"discarding incomplete step_stats tail starting at byte {record_start:,}"
                )
                break

            record_iter = int(record.get("iter", -1))
            if record_iter != expected_iter:
                raise ValueError(
                    f"step statistics are not contiguous: expected iter {expected_iter}, "
                    f"found {record_iter}"
                )
            expected_iter += 1

            if optimizer_name == "norm_control_adam" and recovered_reference_norms is None:
                recovered_reference_norms, recovered_reference_iter = (
                    reference_norms_from_controlled_record(record)
                )

            if record_iter == checkpoint_iter:
                checkpoint_record = record
                checkpoint_end_offset = f.tell()
                break

    if checkpoint_record is None:
        raise ValueError(
            f"step statistics end before checkpoint iter {checkpoint_iter}"
        )
    if int(checkpoint_record.get("tokens", -1)) != int(checkpoint_train_pos):
        raise ValueError(
            f"step_stats tokens at iter {checkpoint_iter} do not match checkpoint train_pos"
        )
    if checkpoint_end_offset < file_size:
        discarded_bytes = file_size - checkpoint_end_offset
        print(
            f"truncating step_stats after iter {checkpoint_iter}: "
            f"discarding {discarded_bytes:,} bytes"
        )
        atomic_copy_prefix(step_stats_path, checkpoint_end_offset)

    return {
        "header": header,
        "checkpoint_record": checkpoint_record,
        "reference_norms": recovered_reference_norms,
        "reference_iter": recovered_reference_iter,
    }


def reconcile_eval_history_for_resume(checkpoint_iter, checkpoint_eval, checkpoint_meta):
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"missing resume eval history: {eval_path}")
    try:
        with open(eval_path, "rb") as f:
            payload = pickle.load(f)
            if f.read(1):
                raise ValueError(f"unexpected trailing bytes in {eval_path}")
    except Exception as exc:
        raise ValueError(f"cannot read eval history: {eval_path}") from exc

    validate_log_meta(payload.get("meta", {}), checkpoint_meta, "eval")
    records = payload.get("eval")
    if not isinstance(records, list):
        raise ValueError(f"eval history has no record list: {eval_path}")
    kept_records = []
    previous_iter = -1
    for record in records:
        record_iter = int(record.get("iter", -1))
        if record_iter <= previous_iter:
            raise ValueError("eval history iterations are not strictly increasing")
        previous_iter = record_iter
        if record_iter <= checkpoint_iter:
            kept_records.append(record)

    if not kept_records or kept_records[-1] != checkpoint_eval:
        raise ValueError("eval history does not end with the checkpoint eval record")
    if len(kept_records) != len(records):
        print(
            f"truncating eval history after iter {checkpoint_iter}: "
            f"discarding {len(records) - len(kept_records)} records"
        )
        atomic_pickle_dump({"meta": payload["meta"], "eval": kept_records}, eval_path)
    return kept_records


def optimizer_resume_state_dict(optimizer):
    if optimizer_name == "adamw":
        return {"optimizer_name": "adamw"}
    if optimizer_name != "norm_control_adam":
        return None
    reference_norms = optimizer.reference_norms
    return {
        "optimizer_name": "norm_control_adam",
        "reference_norms": None if reference_norms is None else dict(reference_norms),
        "reference_iter": optimizer.reference_iter,
        "current_iter": optimizer.current_iter,
        "last_norm_control": dict(optimizer.last_norm_control),
    }


def restore_optimizer_resume_state(optimizer, checkpoint, resume_log_state):
    saved = checkpoint.get("optimizer_resume")
    if optimizer_name == "adamw":
        if saved is not None and saved.get("optimizer_name") != "adamw":
            raise ValueError("checkpoint optimizer_resume state is not for AdamW")
        return

    if saved is not None:
        if saved.get("optimizer_name") != "norm_control_adam":
            raise ValueError("checkpoint optimizer_resume state is not for norm-control Adam")
        reference_norms = saved.get("reference_norms")
        reference_iter = saved.get("reference_iter")
        last_control = saved.get("last_norm_control") or {"active": False}
    else:
        reference_norms = resume_log_state["reference_norms"]
        reference_iter = resume_log_state["reference_iter"]
        last_control = resume_log_state["checkpoint_record"].get(
            "optimizer_control", {"active": False}
        )

    if reference_norms is None:
        if last_control.get("active", False):
            raise ValueError("active norm control has no recoverable reference norms")
        optimizer.reference_norms = None
        optimizer.reference_iter = reference_iter
        optimizer.current_iter = int(checkpoint["iter_num"])
        optimizer._pending_target_ratio = None
        optimizer._pending_active = False
        optimizer.last_norm_control = dict(last_control)
        return
    expected_names = {name for name, _ in optimizer.named_matrix_params}
    if set(reference_norms) != expected_names:
        missing = sorted(expected_names - set(reference_norms))
        unexpected = sorted(set(reference_norms) - expected_names)
        raise ValueError(
            f"norm-control reference names mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    optimizer.reference_norms = {
        name: float(reference_norms[name]) for name in expected_names
    }
    optimizer.reference_iter = reference_iter
    optimizer.current_iter = int(checkpoint["iter_num"])
    optimizer._pending_target_ratio = None
    optimizer._pending_active = False
    optimizer.last_norm_control = dict(last_control)


def validate_optimizer_steps_after_resume(optimizer, checkpoint_iter):
    expected_step = checkpoint_iter + 1
    steps = []
    for state in optimizer.state.values():
        step = state.get("step")
        if step is None:
            continue
        steps.append(int(step.item()) if torch.is_tensor(step) else int(step))
    if not steps or any(step != expected_step for step in steps):
        unique_steps = sorted(set(steps))
        raise ValueError(
            f"optimizer state steps {unique_steps} do not match {expected_step} "
            "completed updates"
        )


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


parameter_infos = list(iter_unique_parameters(raw_model))
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


fresh_run_meta = {
    "format_version": 1,
    "model_name": model_name,
    "dataset": dataset,
    "out_dir": out_dir,
    "data_dir": data_dir,
    "model_config_source": model_config_source,
    "tokens_per_iter": tokens_per_iter,
    "batch_size": batch_size,
    "block_size": block_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "max_iters": max_iters,
    "data_meta": meta,
    "config": config,
    "model_args": model_args,
}
if init_from == "resume":
    run_meta = checkpoint["meta"]
    checkpoint_config_for_save = checkpoint["config"]
else:
    run_meta = fresh_run_meta
    checkpoint_config_for_save = config
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


def save_checkpoint(eval_record):
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": eval_record["iter"],
        "train_pos": train_pos,
        "config": checkpoint_config_for_save,
        "meta": run_meta,
        "eval": eval_record,
        "optimizer_resume": optimizer_resume_state_dict(optimizer),
        "scaler": scaler.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if device_type == "cuda" else None,
        "numpy_rng_state": np.random.get_state(),
    }
    atomic_torch_save(checkpoint, ckpt_path)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, out_key in [("train", "eval_train"), ("val", "eval_val")]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == "train":
                start = (k * batch_tokens) % max(1, len(train_data) - batch_tokens - 1)
                X, Y = make_batch(train_data, start)
            else:
                X, Y = get_val_batch(k)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[out_key] = losses.mean().item()
    model.train()
    return out


def evaluate_and_save(eval_iter):
    losses = estimate_loss()
    record = {
        "iter": int(eval_iter),
        "eval_train": float(losses["eval_train"]),
        "eval_val": float(losses["eval_val"]),
    }
    print(
        f"eval {eval_iter}: eval_train {record['eval_train']:.4f}, "
        f"eval_val {record['eval_val']:.4f}"
    )
    save_eval_record(record)
    print(f"saving checkpoint to {out_dir}")
    save_checkpoint(record)
    return record


if init_from == "scratch":
    initialize_step_stats_file()
    running_mfu = -1.0
else:
    resume_log_state = reconcile_step_stats_for_resume(
        completed_checkpoint_iter,
        train_pos,
        checkpoint["meta"],
    )
    eval_history.extend(
        reconcile_eval_history_for_resume(
            completed_checkpoint_iter,
            checkpoint["eval"],
            checkpoint["meta"],
        )
    )
    restore_optimizer_resume_state(optimizer, checkpoint, resume_log_state)
    validate_optimizer_steps_after_resume(optimizer, completed_checkpoint_iter)
    running_mfu = float(resume_log_state["checkpoint_record"].get("mfu", -1.0))
    torch.set_rng_state(checkpoint["rng_state"])
    if device_type == "cuda":
        if checkpoint["cuda_rng_state"] is None:
            raise ValueError("CUDA resume requires cuda_rng_state in the checkpoint")
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device=device)
    np.random.set_state(checkpoint["numpy_rng_state"])
    print(
        f"resume state is consistent through iter {completed_checkpoint_iter}; "
        f"continuing at iter {iter_num}, train_pos {train_pos:,}"
    )
    checkpoint = None

t0 = time.time()
local_iter_num = 0
while iter_num < max_iters:
    lr = learning_rate_schedule(iter_num)
    gamma_lr = gamma_lr_for_iter(iter_num, lr)
    set_optimizer_learning_rates(optimizer, lr, gamma_lr)
    prepare_optimizer_step(optimizer, iter_num)

    if eval_only:
        if master_process:
            evaluate_and_save(iter_num)
        break

    step_loss_sum = 0.0
    for _ in range(gradient_accumulation_steps):
        X, Y = get_train_batch()
        with ctx:
            _, loss = model(X, Y)
            step_loss_sum += loss.detach().float().item()
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        total_grad_norm = float(total_grad_norm.item())
    else:
        total_grad_norm = None

    train_loss = step_loss_sum / gradient_accumulation_steps
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
    attach_post_step_parameter_stats(
        parameter_stats,
        parameter_infos,
        save_spectral_norms=save_spectral_norms,
        save_row_col_l2_norms=save_row_col_l2_norms,
        spectral_norm_power_iters=spectral_norm_power_iters,
        norm_eps=norm_eps,
    )
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if local_iter_num >= 5:
        mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
    if master_process:
        step_record = {
            "iter": int(iter_num),
            "train_loss": float(train_loss),
            "lr": float(lr),
            "gamma_lr": float(gamma_lr),
            "tokens": int(train_pos),
            "time_sec": float(dt),
            "mfu": float(running_mfu),
            "global_grad_norm": total_grad_norm,
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

# if ddp:
#     destroy_process_group()

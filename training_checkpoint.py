"""Checkpoint state and log recovery shared by fresh and resumed runs."""

import hashlib
import math
import os
import pickle
import random

import numpy as np
import torch
import torch.distributed as dist

from optimizer import AdamH, NormControlMixin


def file_digest(path, byte_count=None):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while byte_count is None or byte_count > 0:
            chunk_size = 16 * 1024 * 1024
            if byte_count is not None:
                chunk_size = min(chunk_size, byte_count)
            chunk = f.read(chunk_size)
            if not chunk:
                if byte_count is not None:
                    raise EOFError(f"{path} ended while checking the dataset")
                break
            digest.update(chunk)
            if byte_count is not None:
                byte_count -= len(chunk)
    return digest.hexdigest()


def dataset_signature(data_dir, train_tokens, val_tokens, train_prefix_tokens, itemsize):
    # The training and eval-train batches only access this prefix. The file
    # lengths also matter because evaluation wraps its offsets at the end.
    return {
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_prefix_tokens": train_prefix_tokens,
        "train_sha256": file_digest(os.path.join(data_dir, "train.bin"), train_prefix_tokens * itemsize),
        "val_sha256": file_digest(os.path.join(data_dir, "val.bin"), val_tokens * itemsize),
    }


def freeze_reference_norm_schedule(schedule, meta, max_iters, checkpoint, rank, world_size):
    if not isinstance(meta, dict) or "cache_file_inside_out_dir" not in meta:
        return schedule, None

    # Older WD=0 helpers lazily write a shared cache. Build it on rank 0 only,
    # then keep the resulting targets with the checkpoint, not the source run.
    if checkpoint is None:
        values = [tuple(schedule(step) for step in range(max_iters)) if rank == 0 else None]
        if world_size > 1:
            dist.broadcast_object_list(values, src=0)
        targets = values[0]
    else:
        targets = checkpoint.get("reference_norm_schedule")
        if targets is None:
            raise ValueError("checkpoint predates saving the WD=0 reference norm schedule; cannot verify its targets")
    if len(targets) != max_iters:
        raise ValueError("reference norm schedule length does not match max_iters")

    def frozen_schedule(step):
        return meta if step == "meta" else targets[step]

    return frozen_schedule, targets


def _hyperball_parameters(optimizer, model):
    hyperball_ids = {
        id(param) for group in optimizer.param_groups if group.get("hyperball", False)
        for param in group["params"]
    }
    return {name: id(param) for name, param in model.named_parameters() if id(param) in hyperball_ids}


def optimizer_extra_state(optimizer, model, optimizer_name):
    saved = {"optimizer_name": optimizer_name}
    if isinstance(optimizer, NormControlMixin):
        norms = optimizer.reference_norms
        saved.update(
            reference_norms=None if norms is None else dict(norms),
            reference_iter=optimizer.reference_iter,
            current_iter=optimizer.current_iter,
            last_norm_control=dict(optimizer.last_norm_control),
        )
    if isinstance(optimizer, AdamH):
        norms = optimizer.reference_norms
        saved.update(
            # Python object IDs change when the model is rebuilt.
            reference_norms=None if norms is None else {
                name: norms[param_id] for name, param_id in _hyperball_parameters(optimizer, model).items()
            },
            reference_iter=optimizer.reference_iter,
            current_iter=optimizer.current_iter,
        )
    if hasattr(optimizer, "last_optimizer_control"):
        saved["last_optimizer_control"] = dict(optimizer.last_optimizer_control)
    return saved


def restore_optimizer_extra(optimizer, model, optimizer_name, checkpoint):
    saved = checkpoint.get("optimizer_resume") or {}
    if saved.get("optimizer_name", optimizer_name) != optimizer_name:
        raise ValueError("checkpoint optimizer metadata does not match the configured optimizer")

    norm_control = isinstance(optimizer, NormControlMixin)
    hyperball = isinstance(optimizer, AdamH)
    if norm_control or hyperball:
        if "reference_norms" not in saved:
            raise ValueError(f"checkpoint does not contain the original {optimizer_name} reference norms")
        if saved.get("current_iter") != checkpoint["iter_num"]:
            raise ValueError("optimizer control state does not match the checkpoint iteration")
        norms = saved["reference_norms"]
        if norm_control:
            parameters = {name: name for name, _ in optimizer.named_matrix_params}
            optimizer.last_norm_control = dict(saved["last_norm_control"])
        else:
            parameters = _hyperball_parameters(optimizer, model)
        if norms is None:
            if hyperball or optimizer.last_norm_control.get("active", False):
                raise ValueError(f"checkpoint has no reference norms for active {optimizer_name}")
        elif set(norms) != set(parameters) or any(not math.isfinite(v) or v < 0.0 for v in norms.values()):
            raise ValueError("checkpoint reference norms do not match the controlled parameters")
        optimizer.reference_norms = None if norms is None else {key: norms[name] for name, key in parameters.items()}
        optimizer.reference_iter = saved["reference_iter"]
        optimizer.current_iter = saved["current_iter"]
        # prepare_step() evaluates the schedule again for the next update.
    if "last_optimizer_control" in saved:
        optimizer.last_optimizer_control = dict(saved["last_optimizer_control"])


def gather_rng_states(device, rank, world_size):
    local_state = {
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if world_size == 1:
        return [local_state]
    states = [None] * world_size if rank == 0 else None
    dist.gather_object(local_state, states, dst=0)
    return states


def restore_rng_state(checkpoint, device, rank, world_size, *, allow_world_size_change=False):
    previous_world_size = checkpoint["meta"].get("world_size", 1)
    if previous_world_size != world_size:
        if not allow_world_size_change:
            raise ValueError("cannot restore per-rank RNG state with a different WORLD_SIZE")
        # The old rank streams have no one-to-one mapping to the new ones.
        # Derive a distinct seed per rank that is stable when retrying this resume.
        run_id = checkpoint["meta"].get("run_id", "")
        key = f"world-size-resume:{run_id}:{checkpoint['iter_num']}:{world_size}"
        base_seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
        seed = (base_seed + rank) % (1 << 64)
        random.seed(seed)
        np.random.seed(seed % (1 << 32))
        torch.manual_seed(seed)
        return

    states = checkpoint.get("rng_by_rank")
    if states is None:
        if world_size != 1:
            raise ValueError("checkpoint predates per-rank RNG saving; cannot exactly resume this DDP run")
        state = checkpoint  # Older single-GPU checkpoints stored RNG fields at the top level.
    else:
        if len(states) != world_size:
            raise ValueError("checkpoint RNG state count does not match WORLD_SIZE")
        state = states[rank]
        random.setstate(state["python_rng_state"])
    torch.set_rng_state(state["rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    if torch.device(device).type == "cuda":
        if state.get("cuda_rng_state") is None:
            raise ValueError(f"checkpoint has no CUDA RNG state for rank {rank}")
        torch.cuda.set_rng_state(state["cuda_rng_state"], device=device)


def validate_checkpoint(checkpoint, config, run_meta, model_args, *, allow_world_size_change=False):
    old_meta = checkpoint["meta"]
    old_config = checkpoint["config"]
    # Device indices and the output directory can change when a run is copied.
    # A different device type is unsupported even when changing world size.
    if old_config.get("simple_training", False) != config.get("simple_training", False):
        raise ValueError("simple_training must match the original run when resuming")
    ignored = {"out_dir", "init_from", "device", "simple_training"}
    for key in sorted((old_config.keys() | config.keys()) - ignored):
        if old_config.get(key) != config.get(key):
            raise ValueError(f"resume config mismatch for {key}: {old_config.get(key)!r} != {config.get(key)!r}")
    if checkpoint["model_args"] != model_args:
        raise ValueError("current model configuration does not match checkpoint model_args")
    if old_meta.get("world_size", 1) != run_meta["world_size"] and not allow_world_size_change:
        raise ValueError("resume requires the same WORLD_SIZE; pass --allow-world-size-change to opt in")
    old_device_type = old_meta.get("device_type", torch.device(old_config["device"]).type)
    if old_device_type != run_meta["device_type"]:
        raise ValueError("resume requires the same device type as the original run")
    for key in ("data_meta", "tokens_per_iter", "config_file_digests", "config_overrides"):
        if key in old_meta and key in run_meta and old_meta[key] != run_meta[key]:
            raise ValueError(f"checkpoint {key} does not match this run")

    completed_iter = checkpoint["iter_num"]
    if not isinstance(completed_iter, int) or not 0 <= completed_iter < run_meta["max_iters"]:
        raise ValueError(f"invalid completed iteration in checkpoint: {completed_iter!r}")
    if checkpoint["eval"].get("iter") != completed_iter:
        raise ValueError("checkpoint eval record does not match its iteration")
    if checkpoint["train_pos"] != (completed_iter + 1) * run_meta["tokens_per_iter"]:
        raise ValueError("checkpoint train_pos does not match its completed updates")


def _validate_log_meta(log_meta, checkpoint_meta, name):
    for key in ("model_name", "dataset", "tokens_per_iter"):
        if log_meta.get(key) != checkpoint_meta[key]:
            raise ValueError(f"{name} does not belong to the checkpoint ({key} differs)")
    # Logs from before run IDs were introduced can still be resumed in place.
    if "run_id" in log_meta and log_meta["run_id"] != checkpoint_meta.get("run_id"):
        raise ValueError(f"{name} does not belong to the checkpoint (run_id differs)")


def read_resume_logs(step_path, eval_path, checkpoint):
    """Validate both logs before touching either one. Ignore any torn tail."""
    completed_iter = checkpoint["iter_num"]
    with open(step_path, "rb") as f:
        header = pickle.load(f)
        if header.get("format") != "pickle_stream" or header.get("record_type") != "step_stats":
            raise ValueError(f"unexpected step statistics format: {step_path}")
        _validate_log_meta(header["meta"], checkpoint["meta"], "step_stats.pkl")
        for expected_iter in range(completed_iter + 1):
            try:
                record = pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as exc:
                raise ValueError(f"step statistics are incomplete before checkpoint iter {completed_iter}") from exc
            if record.get("iter") != expected_iter:
                raise ValueError(f"step statistics are not contiguous at iter {expected_iter}")
        step_end = f.tell()
    if record.get("tokens") != checkpoint["train_pos"]:
        raise ValueError("step statistics token count does not match the checkpoint")

    with open(eval_path, "rb") as f:
        payload = pickle.load(f)
    _validate_log_meta(payload["meta"], checkpoint["meta"], "eval.pkl")
    history = payload["eval"]
    if any(left["iter"] >= right["iter"] for left, right in zip(history, history[1:])):
        raise ValueError("eval history iterations are not strictly increasing")
    kept = [entry for entry in history if entry["iter"] <= completed_iter]
    if not kept or kept[-1] != checkpoint["eval"]:
        raise ValueError("eval history is missing the checkpoint's eval record")
    expected_iters = list(range(0, completed_iter + 1, checkpoint["config"]["eval_interval"]))
    if expected_iters[-1] != completed_iter:
        expected_iters.append(completed_iter)
    if [entry["iter"] for entry in kept] != expected_iters:
        raise ValueError("eval history is incomplete before the checkpoint")
    return step_end, kept, len(history) - len(kept), float(record.get("mfu", -1.0))


def trim_step_log(path, end):
    if os.path.getsize(path) > end:
        with open(path, "r+b") as f:
            f.truncate(end)
            f.flush()
            os.fsync(f.fileno())

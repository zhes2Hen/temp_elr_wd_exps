"""Export a completed run's training logs to one pandas pickle.

Usage: python export_elr_data.py RUN_DIR [OUTPUT_PKL]
Paths are relative to the current working directory. The default output is
../out_data/<run directory name>.pkl. Only use logs from trusted runs.
"""

import os
from pathlib import Path
import pickle
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


def file_signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def read_training_log(path):
    with path.open("rb", buffering=8 * 1024 * 1024) as stream:
        header = pickle.load(stream)
        if header.get("format") != "pickle_stream" or header.get("record_type") != "step_stats":
            raise ValueError(f"unexpected step statistics format: {path}")
        meta = header["meta"]
        count = meta["max_iters"]
        tokens_per_iter = meta["tokens_per_iter"]
        if count <= 0 or tokens_per_iter <= 0:
            raise ValueError("invalid training step or token count in step_stats.pkl")

        steps = np.arange(count, dtype=np.int64)
        lr = np.empty(count, dtype=np.float64)
        elr = np.empty_like(lr)
        train_loss = np.empty_like(lr)
        for step in range(count):
            try:
                row = pickle.load(stream)
            except (EOFError, pickle.UnpicklingError) as exc:
                raise ValueError(f"training log is incomplete or corrupt at step {step}") from exc
            if row["iter"] != step or row["tokens"] != (step + 1) * tokens_per_iter:
                raise ValueError(f"unexpected iteration or token count at step {step}")

            matrices = [p for p in row["parameters"] if p["ndim"] == 2]
            scope = [(p["name"], p["numel"]) for p in matrices]
            if step == 0:
                matrix_scope = scope
                numel = np.asarray([size for _, size in scope], dtype=np.int64)
                if not scope or len({name for name, _ in scope}) != len(scope) or np.any(numel <= 0):
                    raise ValueError("invalid set of two-dimensional parameters")
                weights = numel / numel.sum()
                sqrt_numel = np.sqrt(numel)
            elif scope != matrix_scope:
                raise ValueError(f"two-dimensional parameter scope changed at step {step}")

            matrix_lr = np.asarray([p["lr"] for p in matrices], dtype=np.float64)
            # Match the old export: use the post-update norm and each matrix's actual LR.
            rms = np.asarray([p["parameter"]["fro"] for p in matrices], dtype=np.float64) / sqrt_numel
            if not np.isfinite(rms).all() or np.any(rms <= 0):
                raise ValueError(f"invalid parameter RMS at step {step}")
            if not np.isfinite(matrix_lr).all() or np.any(matrix_lr < 0):
                raise ValueError(f"invalid matrix learning rate at step {step}")

            lr[step] = row["lr"]
            elr[step] = np.sum(weights * matrix_lr / rms)
            train_loss[step] = row["train_loss"]

        if stream.read(1):
            raise ValueError("training log contains data after the final expected step")

    if not (np.isfinite(lr).all() and np.isfinite(elr).all() and np.isfinite(train_loss).all()):
        raise ValueError("training log contains nonfinite learning rates, ELR, or loss")
    return meta, {"step": steps, "lr": lr, "elr": elr, "train_loss": train_loss}


def read_validation_log(path, train_meta):
    with path.open("rb") as stream:
        payload = pickle.load(stream)
        if stream.read(1):
            raise ValueError(f"unexpected trailing data in {path}")
    eval_meta = payload["meta"]
    for key in ("dataset", "model_args", "max_iters", "tokens_per_iter"):
        if eval_meta[key] != train_meta[key]:
            raise ValueError(f"training and validation metadata differ for {key}")
    if "run_id" in train_meta and "run_id" in eval_meta and train_meta["run_id"] != eval_meta["run_id"]:
        raise ValueError("training and validation logs belong to different runs")

    train_config, eval_config = train_meta["config"], eval_meta["config"]
    for key in ("optimizer_name", "weight_decay", "learning_rate_schedule", "norm_control_schedule", "eval_iters"):
        if train_config.get(key) != eval_config.get(key):
            raise ValueError(f"training and validation configurations differ for {key}")

    count = train_meta["max_iters"]
    interval = train_config["eval_interval"]
    if interval <= 0 or eval_config["eval_interval"] != interval:
        raise ValueError("invalid or inconsistent evaluation interval")
    records = payload["eval"]
    steps = np.asarray([record["iter"] for record in records], dtype=np.int64)
    expected = np.union1d(np.arange(0, count, interval, dtype=np.int64), [count - 1])
    if not np.array_equal(steps, expected):
        raise ValueError("missing, duplicate, or unordered validation measurements")

    values = np.asarray([record["eval_val"] for record in records], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("validation log contains nonfinite loss")
    val_loss = np.full(count, np.nan, dtype=np.float64)
    val_loss[steps] = values
    return val_loss


def write_pickle(frame, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            frame.to_pickle(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_run(source, output):
    step_path = source / "step_stats.pkl"
    eval_path = source / "eval.pkl"
    protected = (step_path, eval_path, source / "ckpt.pt")
    if output.resolve() in {path.resolve() for path in protected}:
        raise ValueError("output must not overwrite a training artifact")

    inputs = (step_path, eval_path)
    before = tuple(file_signature(path) for path in inputs)
    meta, columns = read_training_log(step_path)
    columns["val_loss"] = read_validation_log(eval_path, meta)
    if tuple(file_signature(path) for path in inputs) != before:
        raise ValueError("source logs changed during export; wait for the run to finish")

    frame = pd.DataFrame(columns)
    write_pickle(frame, output)
    print(f"Exported {len(frame):,} steps and {frame['val_loss'].count():,} validation measurements to {output}")


def main(args):
    if args in (["-h"], ["--help"]) or len(args) not in (1, 2):
        print(__doc__.strip())
        return 0 if args in (["-h"], ["--help"]) else 2

    source = Path(os.path.abspath(args[0]))
    output = Path(args[1]) if len(args) == 2 else Path("../out_data") / f"{source.name}.pkl"
    try:
        export_run(source, output)
    except (OSError, EOFError, KeyError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""In-memory loss and ELR history for simple training."""

import numpy as np
import torch


METRIC_COLUMNS = ("step", "lr", "mean_weighted_elr", "mean_elr", "global_elr", "train_loss", "val_loss")


class SimpleTrainingMetrics:
    def __init__(self, max_iters, parameter_infos):
        # The full logging path does not require pandas.
        import pandas as pd

        self.pd = pd
        self.matrices = tuple(info["parameter"] for info in parameter_infos if info["parameter"].ndim == 2)
        if not self.matrices:
            raise ValueError("simple training requires at least one trainable two-dimensional parameter")
        numel = np.array([param.numel() for param in self.matrices], dtype=np.int64)
        self.total_numel = int(numel.sum())
        self.sqrt_numel = np.sqrt(numel)
        self.weights = numel / self.total_numel
        self.values = np.full((max_iters, len(METRIC_COLUMNS) - 1), np.nan, dtype=np.float64)
        self.next_step = 0
        self.evaluated_steps = []

    @torch.no_grad()
    def record_step(self, step, lr, loss_sum, accumulation_steps):
        if step != self.next_step:
            raise ValueError(f"expected metrics for step {self.next_step}, got {step}")
        # Keep the old FP32 Frobenius calculation, but transfer the loss and
        # all matrix norms together instead of synchronizing once per tensor.
        norms = [torch.linalg.vector_norm(param.detach().float()) for param in self.matrices]
        scalars = torch.stack([loss_sum, *norms]).cpu().numpy()
        fro = scalars[1:]
        rms = fro / self.sqrt_numel
        global_rms = np.sqrt(np.sum(fro ** 2) / self.total_numel)

        # A zero RMS leaves that ratio undefined; it must not change training.
        with np.errstate(divide="ignore", invalid="ignore"):
            weighted = np.sum(self.weights * lr / rms)
            mean = np.mean(lr / rms)
            global_elr = lr / global_rms
        self.values[step, :-1] = (
            lr,
            weighted if np.isfinite(weighted) else np.nan,
            mean if np.isfinite(mean) else np.nan,
            global_elr if np.isfinite(global_elr) else np.nan,
            float(scalars[0]) / accumulation_steps,
        )
        self.next_step += 1

    def record_validation(self, step, loss):
        if step != self.next_step - 1 or (self.evaluated_steps and step <= self.evaluated_steps[-1]):
            raise ValueError(f"cannot record validation for step {step}")
        self.values[step, -1] = loss
        self.evaluated_steps.append(step)

    def frame(self, meta):
        columns = {"step": np.arange(self.next_step, dtype=np.int64)}
        columns.update((name, self.values[:self.next_step, i]) for i, name in enumerate(METRIC_COLUMNS[1:]))
        frame = self.pd.DataFrame(columns)
        frame.attrs["meta"] = meta
        frame.attrs["evaluated_steps"] = self.evaluated_steps.copy()
        return frame

    def restore(self, path, checkpoint):
        frame = self.pd.read_pickle(path)
        if not isinstance(frame, self.pd.DataFrame) or tuple(frame.columns) != METRIC_COLUMNS:
            raise ValueError("metrics.pkl has an unexpected column layout")
        meta = frame.attrs.get("meta", {})
        for key in ("run_id", "max_iters", "tokens_per_iter"):
            if meta.get(key) != checkpoint["meta"][key]:
                raise ValueError(f"metrics.pkl does not match the checkpoint ({key})")

        completed = checkpoint["iter_num"]
        count = completed + 1
        if len(frame) < count or not np.array_equal(frame["step"].iloc[:count], np.arange(count)):
            raise ValueError("metrics.pkl does not contain all steps through the checkpoint")
        eval_steps = [step for step in frame.attrs.get("evaluated_steps", []) if step < count]
        interval = checkpoint["config"]["eval_interval"]
        expected = list(range(0, count, interval))
        if expected[-1] != completed:
            expected.append(completed)
        if eval_steps != expected:
            raise ValueError("metrics.pkl is missing validation steps before the checkpoint")

        values = frame.iloc[:count, 1:].to_numpy(dtype=np.float64)
        validation = values[:, -1]
        if not np.array_equal(validation[-1], checkpoint["eval"]["eval_val"], equal_nan=True):
            raise ValueError("metrics.pkl validation loss does not match the checkpoint")
        unmeasured = np.ones(count, dtype=bool)
        unmeasured[eval_steps] = False
        if not np.isnan(validation[unmeasured]).all():
            raise ValueError("metrics.pkl has validation loss on an unevaluated step")

        self.values[:count] = values
        self.next_step = count
        self.evaluated_steps = eval_steps
        return len(frame) > count

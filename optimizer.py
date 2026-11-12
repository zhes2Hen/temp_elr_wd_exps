import inspect
import math

import torch


DEEPSEEK_MUON_FAST_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
DEEPSEEK_MUON_STABLE_COEFFICIENTS = (2.0, -1.5, 0.5)
DEEPSEEK_MUON_FAST_STEPS = 8
DEEPSEEK_MUON_STABLE_STEPS = 2


@torch.no_grad()
def zeropower_via_deepseek_hybrid_newton_schulz(matrix, eps=1e-7):
    """Approximate U V^T using DeepSeek-V4's 8+2 hybrid NS iteration."""

    if matrix.ndim != 2:
        raise ValueError(f"hybrid Newton-Schulz requires a 2D tensor, got ndim={matrix.ndim}")
    if eps <= 0.0:
        raise ValueError(f"hybrid Newton-Schulz epsilon must be positive, got {eps}")

    x = matrix.detach().to(dtype=torch.bfloat16)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    # Frobenius normalization guarantees sigma_max <= 1. The norm itself is
    # evaluated in FP32, while all matrix multiplications intentionally use BF16.
    fro = torch.linalg.vector_norm(x.float()).clamp_min(eps)
    x = (x.float() / fro).to(dtype=torch.bfloat16)

    def ns_step(current, coefficients):
        a, b, c = coefficients
        gram = current @ current.T
        polynomial = b * gram + c * (gram @ gram)
        return a * current + polynomial @ current

    for _ in range(DEEPSEEK_MUON_FAST_STEPS):
        x = ns_step(x, DEEPSEEK_MUON_FAST_COEFFICIENTS)
    for _ in range(DEEPSEEK_MUON_STABLE_STEPS):
        x = ns_step(x, DEEPSEEK_MUON_STABLE_COEFFICIENTS)

    return x.T if transposed else x


def spectral_norm_power_iteration(matrix, power_iters=8, eps=1e-12):
    if matrix is None or matrix.ndim != 2:
        return None
    if not isinstance(power_iters, int) or isinstance(power_iters, bool) or power_iters <= 0:
        raise ValueError(f"power_iters must be a positive integer, got {power_iters!r}")
    if eps <= 0.0:
        raise ValueError(f"spectral norm epsilon must be positive, got {eps}")

    x = matrix.detach()
    if x.is_sparse:
        x = x.to_dense()
    x = x.float()
    if x.numel() == 0:
        return 0.0

    # Deterministic start avoids changing the training RNG state for logging.
    idx = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
    v = torch.sin((idx + 1.0) * 12.9898)
    v = v / (torch.linalg.vector_norm(v) + eps)

    for _ in range(power_iters):
        u = x @ v
        u = u / (torch.linalg.vector_norm(u) + eps)
        v = x.T @ u
        v = v / (torch.linalg.vector_norm(v) + eps)

    return torch.dot(u, x @ v).abs().item()


def tensor_norm_stats(
    tensor,
    include_matrix_shape_stats=False,
    include_spectral=True,
    spectral_norm_power_iters=8,
    norm_eps=1e-12,
):
    if tensor is None:
        return None
    x = tensor.detach()
    if x.is_sparse:
        x = x.to_dense()
    x = x.float()
    fro = torch.linalg.vector_norm(x).item()
    stats = {
        "fro": fro,
        "rms": fro / math.sqrt(x.numel()),
    }
    if x.ndim == 2 and include_spectral:
        stats["spectral"] = spectral_norm_power_iteration(
            x,
            power_iters=spectral_norm_power_iters,
            eps=norm_eps,
        )
        stats["spectral_method"] = "power_iteration"
        stats["spectral_power_iters"] = spectral_norm_power_iters
    if x.ndim == 2 and include_matrix_shape_stats:
        stats["row_l2_mean"] = torch.linalg.vector_norm(x, dim=1).mean().item()
        stats["col_l2_mean"] = torch.linalg.vector_norm(x, dim=0).mean().item()
    return stats


def scale_tensor_norm_stats(stats, scale):
    if stats is None:
        return None
    scaled = dict(stats)
    abs_scale = abs(float(scale))
    for key in ("fro", "rms", "spectral", "row_l2_mean", "col_l2_mean"):
        if key in scaled:
            scaled[key] *= abs_scale
    return scaled


def _split_trainable_parameters(model):
    named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    matrix_params = [(name, p) for name, p in named_params if p.dim() >= 2]
    vector_params = [(name, p) for name, p in named_params if p.dim() < 2]
    return matrix_params, vector_params


def _fused_kwargs(optimizer_cls, device_type):
    fused_available = "fused" in inspect.signature(optimizer_cls).parameters
    return {"fused": True} if fused_available and device_type == "cuda" else {}


def _count_params(named_params):
    return sum(p.numel() for _, p in named_params)


class NormControlMixin:
    """Shared per-tensor Frobenius norm projection for norm-control optimizers."""

    def _init_norm_control(self, named_matrix_params, norm_schedule):
        self.named_matrix_params = list(named_matrix_params)
        self.norm_schedule = norm_schedule
        self.reference_norms = None
        self.reference_iter = None
        self.current_iter = None
        self._pending_target_ratio = None
        self._pending_active = False
        self.require_fixed_initial_norms = False
        self.last_norm_control = {"active": False}

    def _schedule_value(self, iter_num):
        value = self.norm_schedule(iter_num)
        if value is None:
            return None
        if isinstance(value, dict):
            if not value.get("active", True):
                return None
            value = value["target_ratio"]
        value = float(value)
        if value < 0.0:
            raise ValueError(f"norm control target ratio must be non-negative, got {value}")
        return value

    @torch.no_grad()
    def _capture_reference_norms(self):
        self.reference_norms = {}
        for name, param in self.named_matrix_params:
            norm = torch.linalg.vector_norm(param.detach().float()).item()
            self.reference_norms[name] = norm
        # current_iter is the first controlled update, so the reference is the
        # parameter state after the previous completed training iteration.
        self.reference_iter = None if self.current_iter is None else self.current_iter - 1

    def prepare_step(self, iter_num):
        self.current_iter = int(iter_num)
        target_ratio = self._schedule_value(self.current_iter)
        if self.require_fixed_initial_norms and target_ratio != 1.0:
            raise ValueError("initial-RMS LR scaling requires a norm ratio of 1 from step 0 onward")
        if target_ratio is None:
            self._pending_active = False
            self._pending_target_ratio = None
            self.last_norm_control = {"active": False, "iter": self.current_iter}
            return

        if self.reference_norms is None:
            self._capture_reference_norms()

        self._pending_active = True
        self._pending_target_ratio = target_ratio

    @torch.no_grad()
    def _project_matrix_parameters(self):
        projected = 0
        skipped_zero = 0
        for name, param in self.named_matrix_params:
            reference_norm = self.reference_norms[name]
            target_norm = self._pending_target_ratio * reference_norm
            current_norm = torch.linalg.vector_norm(param.detach().float()).item()
            if current_norm == 0.0:
                skipped_zero += 1
                continue
            param.mul_(target_norm / current_norm)
            projected += 1

        self.last_norm_control = {
            "active": True,
            "iter": self.current_iter,
            "reference_iter": self.reference_iter,
            "target_ratio": float(self._pending_target_ratio),
            "projected_tensors": projected,
            "skipped_zero_tensors": skipped_zero,
        }


class Muon(torch.optim.Optimizer):
    """
    DeepSeek-hybrid Muon for hidden matrices with an AdamW fallback.

    Muon uses Nesterov momentum, DeepSeek-V4's 8+2 hybrid Newton-Schulz
    orthogonalization, Moonlight's per-matrix update scaling, and decoupled
    weight decay. Parameter groups tagged as AdamW use ordinary AdamW updates.
    """

    def __init__(
        self,
        params,
        lr,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        update_scale=0.2,
        ns_eps=1e-7,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight_decay: {weight_decay}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"invalid Muon momentum: {momentum}")
        if not isinstance(nesterov, bool):
            raise TypeError("Muon nesterov must be a bool")
        if update_scale <= 0.0:
            raise ValueError(f"Muon update_scale must be positive, got {update_scale}")
        if ns_eps <= 0.0:
            raise ValueError(f"Muon ns_eps must be positive, got {ns_eps}")
        beta1, beta2 = adamw_betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"invalid AdamW beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"invalid AdamW beta2: {beta2}")
        if adamw_eps < 0.0:
            raise ValueError(f"invalid AdamW epsilon: {adamw_eps}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "update_scale": update_scale,
            "ns_eps": ns_eps,
            "betas": adamw_betas,
            "eps": adamw_eps,
        }
        super().__init__(params, defaults)
        self.last_optimizer_control = {
            "active": False,
            "type": "muon",
            "orthogonalization": "deepseek_hybrid_newton_schulz_8_plus_2",
        }

    def _muon_direction(self, param, grad, group, *, mutate_state):
        state = self.state[param]
        momentum = group["momentum"]
        momentum_buffer = state.get("momentum_buffer")

        if mutate_state:
            if momentum_buffer is None:
                momentum_buffer = grad.detach().clone(memory_format=torch.preserve_format)
                state["momentum_buffer"] = momentum_buffer
            else:
                momentum_buffer.mul_(momentum).add_(grad)
            next_buffer = momentum_buffer
        elif momentum_buffer is None:
            next_buffer = grad.detach()
        else:
            next_buffer = momentum_buffer.detach().mul(momentum).add(grad)

        if group["nesterov"]:
            return grad.detach().add(next_buffer, alpha=momentum)
        return next_buffer

    @staticmethod
    def _preview_fingerprint(grad, group):
        return (
            id(grad),
            getattr(grad, "_version", None),
            float(group["momentum"]),
            bool(group["nesterov"]),
            float(group["update_scale"]),
            float(group["ns_eps"]),
        )

    @staticmethod
    def _orthogonalized_update(direction, group):
        orthogonal = zeropower_via_deepseek_hybrid_newton_schulz(
            direction,
            eps=group["ns_eps"],
        )
        rows, cols = direction.shape
        moonlight_scale = group["update_scale"] * math.sqrt(max(rows, cols))
        return orthogonal.mul(moonlight_scale)

    @torch.no_grad()
    def preview_muon_update(self, param, grad, group):
        """Return and cache the exact no-LR Muon update used by the next step."""

        if grad.is_sparse:
            raise RuntimeError("Muon does not support sparse gradients")
        if param.ndim != 2 or grad.ndim != 2:
            raise ValueError("Muon parameters and gradients must both be 2D")

        direction = self._muon_direction(param, grad, group, mutate_state=False)
        update = self._orthogonalized_update(direction, group)
        state = self.state[param]
        state["_pending_muon_update"] = update
        state["_pending_muon_fingerprint"] = self._preview_fingerprint(grad, group)
        return update

    @staticmethod
    def _adamw_direction(param, grad, state, beta1, beta2, eps):
        if "exp_avg" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

        state["step"] += 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        step = state["step"]
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        return (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt() + eps)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_tensors = 0
        adamw_tensors = 0
        skipped_tensors = 0
        muon_settings = None
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            optimizer_kind = group.get("optimizer_name", "muon")
            if optimizer_kind == "muon" and muon_settings is None:
                muon_settings = {
                    "momentum": float(group["momentum"]),
                    "nesterov": bool(group["nesterov"]),
                    "update_scale": float(group["update_scale"]),
                    "ns_eps": float(group["ns_eps"]),
                }
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    skipped_tensors += 1
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Muon/AdamW does not support sparse gradients")

                state = self.state[param]
                if optimizer_kind == "muon":
                    direction = self._muon_direction(param, grad, group, mutate_state=True)
                    fingerprint = state.pop("_pending_muon_fingerprint", None)
                    cached_update = state.pop("_pending_muon_update", None)
                    if fingerprint == self._preview_fingerprint(grad, group) and cached_update is not None:
                        update = cached_update
                    else:
                        update = self._orthogonalized_update(direction, group)
                    state["step"] = state.get("step", 0) + 1
                    muon_tensors += 1
                elif optimizer_kind == "adamw":
                    beta1, beta2 = group["betas"]
                    update = self._adamw_direction(param, grad, state, beta1, beta2, group["eps"])
                    adamw_tensors += 1
                else:
                    raise ValueError(f"unsupported optimizer kind inside Muon: {optimizer_kind}")

                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(update, alpha=-lr)

        self.last_optimizer_control = {
            "active": True,
            "type": "muon",
            "orthogonalization": "deepseek_hybrid_newton_schulz_8_plus_2",
            "fast_ns_steps": DEEPSEEK_MUON_FAST_STEPS,
            "stable_ns_steps": DEEPSEEK_MUON_STABLE_STEPS,
            "muon_tensors": muon_tensors,
            "adamw_tensors": adamw_tensors,
            "skipped_tensors": skipped_tensors,
        }
        if muon_settings is not None:
            self.last_optimizer_control.update(muon_settings)
        return loss


class NormControlAdam(NormControlMixin, torch.optim.Adam):
    """
    Adam followed by a per-tensor Frobenius norm projection.

    Only matrix-like parameters are projected. The Adam moments are intentionally
    left unchanged, matching the old train_compose.py implementation style.
    """

    def __init__(self, param_groups, named_matrix_params, norm_schedule, *args, **kwargs):
        super().__init__(param_groups, *args, **kwargs)
        self._init_norm_control(named_matrix_params, norm_schedule)

    def step(self, closure=None):
        loss = super().step(closure)
        if self._pending_active:
            self._project_matrix_parameters()
        return loss


class AdamH(torch.optim.Optimizer):
    """
    AdamH for matrix parameters and Adam for the remaining parameters.

    For each matrix parameter W, the Adam-preconditioned direction U is scaled
    to length lr * ||W||_F and then projected back to the current Frobenius
    sphere. Non-matrix parameters use the same Adam moments and a standard Adam
    step with their own parameter-group learning rate.
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, hyperball_eps=1e-10, norm_schedule=None):
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"invalid epsilon value: {eps}")
        if hyperball_eps <= 0.0:
            raise ValueError(f"invalid hyperball_eps: {hyperball_eps}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"invalid beta2: {beta2}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "hyperball_eps": hyperball_eps,
            "weight_decay": 0.0,
            "hyperball": False,
        }
        super().__init__(params, defaults)
        self.norm_schedule = norm_schedule
        self.reference_norms = None
        self.reference_iter = -1
        self.current_iter = None
        self._pending_target_ratio = None
        self.last_optimizer_control = {"active": False}
        # Enabled only on the rank writing simple Hyperball metrics.
        self.update_direction_norms = None

    @staticmethod
    def _adam_direction(param, grad, state, beta1, beta2, eps):
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        state["step"] += 1
        step = state["step"]

        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        return (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt() + eps)

    def prepare_step(self, iter_num):
        self.current_iter = int(iter_num)
        self._ensure_reference_norms()
        self._pending_target_ratio = self._schedule_value(self.current_iter)

    def _schedule_value(self, iter_num):
        if self.norm_schedule is None:
            return 1.0
        value = self.norm_schedule(iter_num)
        if value is None:
            return 1.0
        if isinstance(value, dict):
            if not value.get("active", True):
                return 1.0
            value = value["target_ratio"]
        value = float(value)
        if value < 0.0:
            raise ValueError(f"AdamH target radius ratio must be non-negative, got {value}")
        return value

    @torch.no_grad()
    def _ensure_reference_norms(self):
        if self.reference_norms is not None:
            return
        self.reference_norms = {}
        for group in self.param_groups:
            if not group.get("hyperball", False):
                continue
            for param in group["params"]:
                self.reference_norms[id(param)] = torch.linalg.vector_norm(param.detach().float()).item()

    def target_hyperball_norm(self, param):
        self._ensure_reference_norms()
        if self._pending_target_ratio is None:
            iter_num = 0 if self.current_iter is None else self.current_iter
            self._pending_target_ratio = self._schedule_value(iter_num)
        return self.reference_hyperball_norm(param) * self._pending_target_ratio

    def reference_hyperball_norm(self, param):
        self._ensure_reference_norms()
        return self.reference_norms[id(param)]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        hyperball_tensors = 0
        adam_tensors = 0
        skipped_zero_tensors = 0
        if self.update_direction_norms is not None:
            self.update_direction_norms.clear()
        self._ensure_reference_norms()
        if self._pending_target_ratio is None:
            iter_num = 0 if self.current_iter is None else self.current_iter
            self._pending_target_ratio = self._schedule_value(iter_num)
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            hyperball_eps = group["hyperball_eps"]
            use_hyperball = group.get("hyperball", False)
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AdamH does not support sparse gradients")

                direction = self._adam_direction(param, param.grad, self.state[param], beta1, beta2, eps)
                if use_hyperball:
                    reference_norm = self.reference_hyperball_norm(param)
                    if reference_norm == 0.0:
                        skipped_zero_tensors += 1
                        continue

                    target_norm = self.target_hyperball_norm(param)
                    direction_norm = torch.linalg.vector_norm(direction.detach().float()).item()
                    if self.update_direction_norms is not None:
                        self.update_direction_norms[id(param)] = direction_norm
                    step_scale = target_norm / max(direction_norm, hyperball_eps)
                    param.add_(direction * step_scale, alpha=-lr)

                    current_norm = torch.linalg.vector_norm(param.detach().float()).item()
                    if current_norm == 0.0:
                        skipped_zero_tensors += 1
                        continue
                    param.mul_(target_norm / current_norm)
                    hyperball_tensors += 1
                else:
                    param.add_(direction, alpha=-lr)
                    adam_tensors += 1

        self.last_optimizer_control = {
            "active": True,
            "iter": self.current_iter,
            "type": "adamh",
            "reference_iter": self.reference_iter,
            "target_ratio": float(self._pending_target_ratio),
            "projected_tensors": hyperball_tensors,
            "hyperball_tensors": hyperball_tensors,
            "adam_tensors": adam_tensors,
            "skipped_zero_tensors": skipped_zero_tensors,
        }
        return loss


class MuonH(AdamH):
    """
    MuonH for hidden matrices, AdamH for embeddings/heads, and Adam for vectors.

    MuonH first applies Nesterov momentum and DeepSeek's 8+2 hybrid
    Newton-Schulz iteration. The resulting matrix is Frobenius-normalized,
    scaled to the scheduled Hyperball radius, applied with the group learning
    rate, and followed by a projection back to that radius.
    """

    def __init__(
        self,
        params,
        lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        hyperball_eps=1e-10,
        norm_schedule=None,
        momentum=0.95,
        nesterov=True,
        ns_eps=1e-7,
    ):
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"invalid MuonH momentum: {momentum}")
        if not isinstance(nesterov, bool):
            raise TypeError("MuonH nesterov must be a bool")
        if ns_eps <= 0.0:
            raise ValueError(f"MuonH ns_eps must be positive, got {ns_eps}")

        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            hyperball_eps=hyperball_eps,
            norm_schedule=norm_schedule,
        )
        for group in self.param_groups:
            group.setdefault("momentum", momentum)
            group.setdefault("nesterov", nesterov)
            group.setdefault("ns_eps", ns_eps)
        self.muon_momentum = momentum
        self.muon_nesterov = nesterov
        self.last_optimizer_control = {
            "active": False,
            "type": "muonh",
            "orthogonalization": "deepseek_hybrid_newton_schulz_8_plus_2",
        }

    def _muon_direction(self, param, grad, group, *, mutate_state):
        state = self.state[param]
        momentum = group["momentum"]
        momentum_buffer = state.get("momentum_buffer")

        if mutate_state:
            if momentum_buffer is None:
                momentum_buffer = grad.detach().clone(memory_format=torch.preserve_format)
                state["momentum_buffer"] = momentum_buffer
            else:
                momentum_buffer.mul_(momentum).add_(grad)
            next_buffer = momentum_buffer
        elif momentum_buffer is None:
            next_buffer = grad.detach()
        else:
            next_buffer = momentum_buffer.detach().mul(momentum).add(grad)

        if group["nesterov"]:
            return grad.detach().add(next_buffer, alpha=momentum)
        return next_buffer

    def _preview_fingerprint(self, grad, group):
        return (
            id(grad),
            getattr(grad, "_version", None),
            float(group["momentum"]),
            bool(group["nesterov"]),
            float(group["ns_eps"]),
            float(self._pending_target_ratio),
        )

    @staticmethod
    def _orthogonalized_update(direction, group):
        # Hyperball normalizes this result in Frobenius norm, so any positive
        # Moonlight per-shape scale would cancel exactly here.
        return zeropower_via_deepseek_hybrid_newton_schulz(
            direction,
            eps=group["ns_eps"],
        )

    def _normalize_hyperball_update(self, param, update, group):
        reference_norm = self.reference_hyperball_norm(param)
        if reference_norm == 0.0:
            return torch.zeros_like(update), 0.0, None

        target_norm = self.target_hyperball_norm(param)
        update_norm = torch.linalg.vector_norm(update.detach().float()).item()
        normalization_scale = target_norm / max(update_norm, group["hyperball_eps"])
        return update * normalization_scale, normalization_scale, update_norm

    @torch.no_grad()
    def preview_muonh_update(self, param, grad, group):
        """Return the exact MuonH proposal and its pre-normalization direction."""

        if grad.is_sparse:
            raise RuntimeError("MuonH does not support sparse gradients")
        if param.ndim != 2 or grad.ndim != 2:
            raise ValueError("MuonH parameters and gradients must both be 2D")

        direction = self._muon_direction(param, grad, group, mutate_state=False)
        pre_normalization = self._orthogonalized_update(direction, group)
        update, normalization_scale, update_norm = self._normalize_hyperball_update(
            param,
            pre_normalization,
            group,
        )
        state = self.state[param]
        state["_pending_muonh_update"] = (update, update_norm)
        state["_pending_muonh_fingerprint"] = self._preview_fingerprint(grad, group)
        return update, pre_normalization, normalization_scale

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muonh_tensors = 0
        adamh_tensors = 0
        adam_tensors = 0
        skipped_zero_tensors = 0
        skipped_no_grad_tensors = 0
        if self.update_direction_norms is not None:
            self.update_direction_norms.clear()
        self._ensure_reference_norms()
        if self._pending_target_ratio is None:
            iter_num = 0 if self.current_iter is None else self.current_iter
            self._pending_target_ratio = self._schedule_value(iter_num)

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            optimizer_kind = group.get("optimizer_name", "muonh")
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    skipped_no_grad_tensors += 1
                    continue
                if grad.is_sparse:
                    raise RuntimeError("MuonH/AdamH/Adam does not support sparse gradients")

                state = self.state[param]
                if optimizer_kind == "muonh":
                    direction = self._muon_direction(param, grad, group, mutate_state=True)
                    fingerprint = state.pop("_pending_muonh_fingerprint", None)
                    cached_update = state.pop("_pending_muonh_update", None)
                    if fingerprint == self._preview_fingerprint(grad, group) and cached_update is not None:
                        update, update_norm = cached_update
                    else:
                        pre_normalization = self._orthogonalized_update(direction, group)
                        update, _, update_norm = self._normalize_hyperball_update(
                            param,
                            pre_normalization,
                            group,
                        )
                    state["step"] = state.get("step", 0) + 1
                    use_hyperball = True
                elif optimizer_kind in {"adamh", "adam"}:
                    update = self._adam_direction(param, grad, state, beta1, beta2, group["eps"])
                    use_hyperball = optimizer_kind == "adamh"
                    if use_hyperball:
                        update, _, update_norm = self._normalize_hyperball_update(param, update, group)
                    else:
                        adam_tensors += 1
                else:
                    raise ValueError(f"unsupported optimizer kind inside MuonH: {optimizer_kind}")

                if use_hyperball:
                    if self.reference_hyperball_norm(param) == 0.0:
                        skipped_zero_tensors += 1
                        continue
                    if self.update_direction_norms is not None:
                        self.update_direction_norms[id(param)] = update_norm
                    target_norm = self.target_hyperball_norm(param)
                    param.add_(update, alpha=-lr)
                    current_norm = torch.linalg.vector_norm(param.detach().float()).item()
                    if current_norm == 0.0:
                        skipped_zero_tensors += 1
                        continue
                    param.mul_(target_norm / current_norm)
                    if optimizer_kind == "muonh":
                        muonh_tensors += 1
                    else:
                        adamh_tensors += 1
                else:
                    param.add_(update, alpha=-lr)

        projected_tensors = muonh_tensors + adamh_tensors
        self.last_optimizer_control = {
            "active": True,
            "iter": self.current_iter,
            "type": "muonh",
            "reference_iter": self.reference_iter,
            "target_ratio": float(self._pending_target_ratio),
            "orthogonalization": "deepseek_hybrid_newton_schulz_8_plus_2",
            "fast_ns_steps": DEEPSEEK_MUON_FAST_STEPS,
            "stable_ns_steps": DEEPSEEK_MUON_STABLE_STEPS,
            "momentum": float(self.muon_momentum),
            "nesterov": bool(self.muon_nesterov),
            "projected_tensors": projected_tensors,
            "muonh_tensors": muonh_tensors,
            "adamh_tensors": adamh_tensors,
            "adam_tensors": adam_tensors,
            "skipped_zero_tensors": skipped_zero_tensors,
            "skipped_no_grad_tensors": skipped_no_grad_tensors,
        }
        return loss


def build_optimizer(
    model,
    optimizer_name,
    weight_decay,
    lr,
    betas,
    device_type,
    adam_eps=1e-8,
    norm_control_schedule=None,
    muon_momentum=0.95,
    muon_nesterov=True,
    muon_update_scale=0.2,
    muon_ns_eps=1e-7,
):
    matrix_params, vector_params = _split_trainable_parameters(model)
    optimizer_name = optimizer_name.lower()

    if optimizer_name in {"adamw", "adamw_lr_only"}:
        optim_groups = [
            {
                "params": [p for _, p in matrix_params],
                "weight_decay": weight_decay,
                "optimizer_name": "adamw",
                "parameter_kind": "matrix",
            },
            {
                "params": [p for _, p in vector_params],
                "weight_decay": 0.0,
                "optimizer_name": "adamw",
                "parameter_kind": "vector",
            },
        ]
        print(f"num decayed parameter tensors: {len(matrix_params)}, with {_count_params(matrix_params):,} parameters")
        print(f"num non-decayed parameter tensors: {len(vector_params)}, with {_count_params(vector_params):,} parameters")
        return torch.optim.AdamW(
            optim_groups,
            lr=lr,
            betas=betas,
            eps=adam_eps,
            **_fused_kwargs(torch.optim.AdamW, device_type),
        )

    if optimizer_name == "muon":
        muon_params = [
            (name, param)
            for name, param in matrix_params
            if param.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        muon_param_ids = {id(param) for _, param in muon_params}
        adamw_matrix_params = [
            (name, param)
            for name, param in matrix_params
            if id(param) not in muon_param_ids
        ]
        optim_groups = [
            {
                "params": [param for _, param in muon_params],
                "weight_decay": weight_decay,
                "optimizer_name": "muon",
                "parameter_kind": "matrix",
                "momentum": muon_momentum,
                "nesterov": muon_nesterov,
                "update_scale": muon_update_scale,
                "ns_eps": muon_ns_eps,
            },
            {
                "params": [param for _, param in adamw_matrix_params],
                "weight_decay": weight_decay,
                "optimizer_name": "adamw",
                "parameter_kind": "matrix",
            },
            {
                "params": [param for _, param in vector_params],
                "weight_decay": 0.0,
                "optimizer_name": "adamw",
                "parameter_kind": "vector",
            },
        ]
        print(f"num Muon hidden-matrix tensors: {len(muon_params)}, with {_count_params(muon_params):,} parameters")
        print(
            f"num AdamW embedding/head matrix tensors: {len(adamw_matrix_params)}, "
            f"with {_count_params(adamw_matrix_params):,} parameters"
        )
        print(f"num AdamW vector tensors: {len(vector_params)}, with {_count_params(vector_params):,} parameters")
        return Muon(
            optim_groups,
            lr=lr,
            weight_decay=weight_decay,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            update_scale=muon_update_scale,
            ns_eps=muon_ns_eps,
            adamw_betas=betas,
            adamw_eps=adam_eps,
        )

    if optimizer_name == "norm_control_adam":
        if norm_control_schedule is None:
            raise ValueError("optimizer_name='norm_control_adam' requires norm_control_schedule")
        if weight_decay != 0.0:
            raise ValueError("norm_control_adam replaces weight decay; set weight_decay=0.0")
        optim_groups = [
            {
                "params": [p for _, p in matrix_params],
                "weight_decay": 0.0,
                "optimizer_name": "norm_control_adam",
                "parameter_kind": "matrix",
            },
            {
                "params": [p for _, p in vector_params],
                "weight_decay": 0.0,
                "optimizer_name": "adam",
                "parameter_kind": "vector",
            },
        ]
        print(f"num norm-controlled matrix tensors: {len(matrix_params)}, with {_count_params(matrix_params):,} parameters")
        print(f"num Adam vector tensors: {len(vector_params)}, with {_count_params(vector_params):,} parameters")
        return NormControlAdam(
            optim_groups,
            matrix_params,
            norm_control_schedule,
            lr=lr,
            betas=betas,
            eps=adam_eps,
            weight_decay=0.0,
            **_fused_kwargs(torch.optim.Adam, device_type),
        )

    if optimizer_name == "adamh":
        if weight_decay != 0.0:
            raise ValueError("adamh keeps matrix norms fixed and uses no decoupled weight decay; set weight_decay=0.0")
        adamh_params = [(name, p) for name, p in matrix_params if p.dim() == 2]
        adam_params = vector_params + [(name, p) for name, p in matrix_params if p.dim() != 2]
        optim_groups = [
            {
                "params": [p for _, p in adamh_params],
                "weight_decay": 0.0,
                "optimizer_name": "adamh",
                "parameter_kind": "matrix",
                "hyperball": True,
            },
            {
                "params": [p for _, p in adam_params],
                "weight_decay": 0.0,
                "optimizer_name": "adam",
                "parameter_kind": "vector",
                "hyperball": False,
            },
        ]
        print(f"num AdamH 2D tensors: {len(adamh_params)}, with {_count_params(adamh_params):,} parameters")
        print(f"num Adam non-2D tensors: {len(adam_params)}, with {_count_params(adam_params):,} parameters")
        return AdamH(
            optim_groups,
            lr=lr,
            betas=betas,
            eps=adam_eps,
            norm_schedule=norm_control_schedule,
        )

    if optimizer_name == "muonh":
        if weight_decay != 0.0:
            raise ValueError("muonh keeps matrix norms fixed and uses no decoupled weight decay; set weight_decay=0.0")
        adamh_params = [
            (name, param)
            for name, param in matrix_params
            if param.ndim == 2 and ("embed_tokens" in name or "lm_head" in name)
        ]
        adamh_param_ids = {id(param) for _, param in adamh_params}
        muonh_params = [
            (name, param)
            for name, param in matrix_params
            if param.ndim == 2 and id(param) not in adamh_param_ids
        ]
        muonh_param_ids = {id(param) for _, param in muonh_params}
        adam_params = vector_params + [
            (name, param)
            for name, param in matrix_params
            if id(param) not in adamh_param_ids and id(param) not in muonh_param_ids
        ]
        optim_groups = [
            {
                "params": [param for _, param in muonh_params],
                "weight_decay": 0.0,
                "optimizer_name": "muonh",
                "parameter_kind": "matrix",
                "hyperball": True,
                "momentum": muon_momentum,
                "nesterov": muon_nesterov,
                "ns_eps": muon_ns_eps,
            },
            {
                "params": [param for _, param in adamh_params],
                "weight_decay": 0.0,
                "optimizer_name": "adamh",
                "parameter_kind": "matrix",
                "hyperball": True,
            },
            {
                "params": [param for _, param in adam_params],
                "weight_decay": 0.0,
                "optimizer_name": "adam",
                "parameter_kind": "vector",
                "hyperball": False,
            },
        ]
        print(f"num MuonH hidden-matrix tensors: {len(muonh_params)}, with {_count_params(muonh_params):,} parameters")
        print(
            f"num AdamH embedding/head tensors: {len(adamh_params)}, "
            f"with {_count_params(adamh_params):,} parameters"
        )
        print(f"num Adam non-2D tensors: {len(adam_params)}, with {_count_params(adam_params):,} parameters")
        return MuonH(
            optim_groups,
            lr=lr,
            betas=betas,
            eps=adam_eps,
            norm_schedule=norm_control_schedule,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_eps=muon_ns_eps,
        )

    raise ValueError(f"unknown optimizer_name: {optimizer_name}")


def prepare_optimizer_step(optimizer, iter_num):
    if hasattr(optimizer, "prepare_step"):
        optimizer.prepare_step(iter_num)


def optimizer_control_state(optimizer):
    if hasattr(optimizer, "last_optimizer_control"):
        return optimizer.last_optimizer_control
    if hasattr(optimizer, "last_norm_control"):
        return optimizer.last_norm_control
    return {"active": False}


def optimizer_state_step(state):
    step = state.get("step", 0)
    if torch.is_tensor(step):
        return int(step.item())
    return int(step)


def optimizer_update_without_lr(
    param,
    group,
    optimizer,
    default_optimizer_name,
    return_pre_normalization=False,
):
    """
    Return the optimizer update direction before multiplying by lr.

    Decoupled weight decay and final norm projections are intentionally not
    included here. For AdamH and MuonH the normal return value is the Hyperball
    proposal R * U / max(||U||, eps). With return_pre_normalization=True, also
    return the unnormalized base direction U and its normalization scale.
    """

    raw_grad = param.grad
    if raw_grad is None:
        result = (None, None, None)
        return result if return_pre_normalization else None

    optimizer_kind = group.get("optimizer_name", default_optimizer_name)
    if optimizer_kind == "muon":
        if not hasattr(optimizer, "preview_muon_update"):
            raise TypeError("Muon statistics require optimizer.preview_muon_update")
        update = optimizer.preview_muon_update(param, raw_grad, group)
        result = (update, None, None)
        return result if return_pre_normalization else update

    if optimizer_kind == "muonh":
        if not hasattr(optimizer, "preview_muonh_update"):
            raise TypeError("MuonH statistics require optimizer.preview_muonh_update")
        result = optimizer.preview_muonh_update(param, raw_grad, group)
        return result if return_pre_normalization else result[0]

    grad = raw_grad.detach().float()
    beta1, beta2 = group["betas"]
    eps = group["eps"]
    state = optimizer.state.get(param, {})
    next_step = optimizer_state_step(state) + 1

    if "exp_avg" in state:
        exp_avg = state["exp_avg"].detach().float()
        exp_avg_sq = state["exp_avg_sq"].detach().float()
        next_exp_avg = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
        next_exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1.0 - beta2)
    else:
        next_exp_avg = grad.mul(1.0 - beta1)
        next_exp_avg_sq = grad.square().mul(1.0 - beta2)

    m_hat = next_exp_avg / (1.0 - beta1**next_step)
    v_hat = next_exp_avg_sq / (1.0 - beta2**next_step)
    adam_direction = m_hat / (v_hat.sqrt() + eps)

    if optimizer_kind == "adamh":
        normalization_scale = 0.0
        if hasattr(optimizer, "reference_hyperball_norm") and optimizer.reference_hyperball_norm(param) == 0.0:
            update = torch.zeros_like(adam_direction)
        else:
            if hasattr(optimizer, "target_hyperball_norm"):
                target_norm = optimizer.target_hyperball_norm(param)
            else:
                target_norm = torch.linalg.vector_norm(param.detach().float()).item()
            direction_norm = torch.linalg.vector_norm(adam_direction.detach().float()).item()
            hyperball_eps = group.get("hyperball_eps", 1e-10)
            normalization_scale = target_norm / max(direction_norm, hyperball_eps)
            update = adam_direction * normalization_scale

        result = (update, adam_direction, normalization_scale)
        return result if return_pre_normalization else update

    result = (adam_direction, None, None)
    return result if return_pre_normalization else adam_direction


def projection_stats(direction, param):
    if direction is None:
        return None
    u = direction.detach().float().reshape(-1)
    w = param.detach().float().reshape(-1)
    direction_fro = torch.linalg.vector_norm(u).item()
    param_fro = torch.linalg.vector_norm(w).item()
    inner = torch.dot(u, w).item()
    if param_fro == 0.0:
        coeff = 0.0
        parallel_fro = 0.0
        orthogonal_fro = direction_fro
    else:
        coeff = inner / (param_fro * param_fro)
        parallel_fro = abs(inner) / param_fro
        orthogonal_fro = max(direction_fro * direction_fro - parallel_fro * parallel_fro, 0.0) ** 0.5
    return {
        "inner_product": inner,
        "coefficient": coeff,
        "parallel_fro": parallel_fro,
        "orthogonal_fro": orthogonal_fro,
    }


@torch.no_grad()
def collect_pre_step_parameter_stats(
    parameter_infos,
    optimizer,
    optimizer_group_by_param_id,
    default_optimizer_name,
    *,
    save_spectral_norms,
    save_row_col_l2_norms,
    spectral_norm_power_iters,
    norm_eps,
):
    """Collect gradient and optimizer-update statistics before optimizer.step()."""

    stats_kwargs = {
        "include_matrix_shape_stats": save_row_col_l2_norms,
        "include_spectral": save_spectral_norms,
        "spectral_norm_power_iters": spectral_norm_power_iters,
        "norm_eps": norm_eps,
    }
    records = []
    for info in parameter_infos:
        param = info["parameter"]
        param_id = id(param)
        group = optimizer_group_by_param_id[param_id]
        optimizer_kind = group.get("optimizer_name", default_optimizer_name)
        update, update_pre_normalization, normalization_scale = optimizer_update_without_lr(
            param,
            group,
            optimizer,
            default_optimizer_name,
            return_pre_normalization=True,
        )
        pre_normalization_stats = tensor_norm_stats(update_pre_normalization, **stats_kwargs)
        if pre_normalization_stats is None:
            update_stats = tensor_norm_stats(update, **stats_kwargs)
        else:
            update_stats = scale_tensor_norm_stats(pre_normalization_stats, normalization_scale)

        record = {
            "name": info["name"],
            "shape": tuple(param.shape),
            "ndim": param.ndim,
            "numel": param.numel(),
            "optimizer": optimizer_kind,
            "parameter_kind": group.get("parameter_kind", "unknown"),
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            # The parameter norm is filled after optimizer.step(), so it reflects
            # the optimizer update plus any norm-control projection.
            "parameter": None,
            "gradient": tensor_norm_stats(param.grad, **stats_kwargs),
            "gradient_proj": projection_stats(param.grad, param),
            "update": update_stats,
            "update_proj": projection_stats(update, param),
        }
        if optimizer_kind in {"adamh", "muonh"} and param.ndim == 2:
            record["update_pre_normalization"] = pre_normalization_stats
        records.append(record)
        del update, update_pre_normalization
    return records


@torch.no_grad()
def attach_post_step_parameter_stats(
    records,
    parameter_infos,
    *,
    save_spectral_norms,
    save_row_col_l2_norms,
    spectral_norm_power_iters,
    norm_eps,
):
    for record, info in zip(records, parameter_infos):
        record["parameter"] = tensor_norm_stats(
            info["parameter"],
            include_matrix_shape_stats=save_row_col_l2_norms,
            include_spectral=save_spectral_norms,
            spectral_norm_power_iters=spectral_norm_power_iters,
            norm_eps=norm_eps,
        )

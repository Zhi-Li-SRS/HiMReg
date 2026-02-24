"""Adaptive Stochastic Gradient Descent optimizer following Elastix's Robbins-Monro schedule.

step_k = a / (A + k + 1)^alpha

where:
  a   — initial step size (auto-estimated from gradient norms if not provided)
  A   — relaxation factor = max_iter / 10
  alpha — decay exponent (default 1.0)
"""

from typing import Optional

import torch
from torch.optim import Optimizer


class AdaptiveStochasticGradientDescent(Optimizer):
    """ASGD with Robbins-Monro decaying step size (Elastix-style).

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize.
    a : float or None
        Initial step size.  When ``None``, automatically estimated from
        gradient norms over the first ``auto_est_steps`` steps.
    A : float or None
        Relaxation factor.  Defaults to ``max_iter / 10``.
    alpha : float
        Decay exponent (default 1.0).
    max_iter : int
        Expected number of iterations at the current pyramid scale
        (used to compute *A* and for schedule shaping).
    auto_est_steps : int
        Number of warm-up steps used to estimate *a* when ``a is None``.
    """

    def __init__(
        self,
        params,
        a: Optional[float] = None,
        A: Optional[float] = None,
        alpha: float = 1.0,
        max_iter: int = 200,
        auto_est_steps: int = 5,
    ):
        defaults = dict(a=a, A=A, alpha=alpha, max_iter=max_iter, auto_est_steps=auto_est_steps)
        super().__init__(params, defaults)

        # Global state shared across param groups.
        self.state.setdefault("step_count", 0)
        self.state.setdefault("grad_norm_accum", 0.0)
        self.state.setdefault("grad_norm_count", 0)
        self.state.setdefault("a_estimated", a is not None)
        self.state.setdefault("a_value", a if a is not None else 1.0)

        self._alpha = alpha
        self._auto_est_steps = auto_est_steps
        self._A = A if A is not None else max(max_iter / 10.0, 1.0)

    def reset(self, max_iter: int) -> None:
        """Reset step counter for a new pyramid scale."""
        self.state["step_count"] = 0
        self.state["grad_norm_accum"] = 0.0
        self.state["grad_norm_count"] = 0
        if not self.state["a_estimated"]:
            # Re-estimate at each new scale if no explicit a was given.
            pass
        self._A = max(max_iter / 10.0, 1.0)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        k = self.state["step_count"]

        # --- Auto-estimation of `a` from gradient norms ---
        if not self.state["a_estimated"]:
            total_norm = 0.0
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            self.state["grad_norm_accum"] += total_norm
            self.state["grad_norm_count"] += 1

            if self.state["grad_norm_count"] >= self._auto_est_steps:
                avg_norm = self.state["grad_norm_accum"] / self.state["grad_norm_count"]
                self.state["a_value"] = 1.0 / max(avg_norm, 1e-8)
                self.state["a_estimated"] = True

        a = self.state["a_value"]
        step_size = a / (self._A + k + 1) ** self._alpha

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.add_(p.grad, alpha=-step_size)

        self.state["step_count"] = k + 1
        return loss

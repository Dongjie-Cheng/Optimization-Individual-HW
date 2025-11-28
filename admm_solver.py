# admm_solver.py
from __future__ import annotations
import time
import numpy as np

try:
    import scipy.sparse as sp
    _HAVE_SP = True
except Exception:
    sp = None
    _HAVE_SP = False

from utils import (
    sigmoid, soft_threshold,
    duality_gap
)

class ADMMConfig:
    def __init__(
        self,
        max_iters: int = 2000,
        tol_gap: float = 1e-4,
        rho_init: float = 1.0,
        alpha: float = 1.5,          # over-relaxation in (1,2)
        adapt_every: int = 10,       # residual balancing frequency
        mu: float = 10.0,            # residual ratio threshold
        eta: float = 2.0,            # rho multiplier/divider
        newton_steps: int = 1,       # inner Newton steps for w-subproblem
        step_scale: float = 1.0,     # damping for diagonal-Newton
        freeze_rho_at: float = 5.0,  # when gap < freeze_rho_at * tol_gap, stop adapting rho
        verbose: bool = False,
    ):
        self.max_iters = max_iters
        self.tol_gap = tol_gap
        self.rho_init = rho_init
        self.alpha = alpha
        self.adapt_every = adapt_every
        self.mu = mu
        self.eta = eta
        self.newton_steps = newton_steps
        self.step_scale = step_scale
        self.freeze_rho_at = freeze_rho_at
        self.verbose = verbose


def _xt_dot(X, v: np.ndarray) -> np.ndarray:
    if _HAVE_SP and sp.issparse(X):
        return X.T @ v
    return X.T.dot(v)


def _x2t_dot(X, s: np.ndarray) -> np.ndarray:
    """
    Compute diag((X^T) diag(s) X) = (X**2)^T s
    Works for dense or sparse CSR/CSC X.
    """
    if _HAVE_SP and sp.issparse(X):
        return np.asarray((X.power(2).T @ s)).ravel()
    else:
        return ((X ** 2).T @ s)


def admm_solve(
    X, y: np.ndarray, lam: float,
    w0: np.ndarray | None = None,
    cfg: ADMMConfig = ADMMConfig()
):
    """
    ADMM for L1-logistic:
      min_w (1/m) sum log(1+exp(x_i^T w)) - y_i x_i^T w + lam ||w||_1
    Split: w=z, with rho adaptive, over-relaxation alpha, and gap-based stop.

    Returns:
      w        : solution
      history  : list of dicts per iteration (gap, P, D, rho, times, residuals)
    """
    t0 = time.perf_counter()
    m, n = X.shape

    # init
    w = np.zeros(n) if w0 is None else w0.copy()
    z = w.copy()
    u = np.zeros(n)
    z_prev = z.copy()

    rho = float(cfg.rho_init)
    adapt_enabled = (cfg.adapt_every is not None) and (cfg.adapt_every > 0)

    history = []
    for it in range(cfg.max_iters):
        # ----- w-update: inexact diagonal-Newton for f(w) + (rho/2)||w - z + u||^2
        for _ in range(cfg.newton_steps):
            zlin = X @ w
            p = sigmoid(zlin)
            grad = (_xt_dot(X, (p - y)) / m) + rho * (w - z + u)
            s = p * (1 - p)
            H_diag = (_x2t_dot(X, s) / m) + rho
            step = grad / np.maximum(H_diag, 1e-8)
            w = w - cfg.step_scale * step

        # ----- over-relaxation
        w_hat = cfg.alpha * w + (1.0 - cfg.alpha) * z

        # ----- z-update: prox l1
        z = soft_threshold(w_hat + u, lam / rho)

        # ----- u-update
        u = u + (w_hat - z)

        # ----- residuals
        r_norm = float(np.linalg.norm(w - z))
        d_norm = float(rho * np.linalg.norm(z - z_prev))

        # ----- duality gap (certificate stop)
        gap, P, D, u_dual, theta = duality_gap(X, y, w, lam)

        # ----- adaptive rho (residual balancing)
        if adapt_enabled and ((it + 1) % cfg.adapt_every == 0):
            if r_norm > cfg.mu * d_norm:
                rho *= cfg.eta
                u /= cfg.eta
            elif d_norm > cfg.mu * r_norm:
                rho /= cfg.eta
                u *= cfg.eta
            z_prev = z.copy()

        # optionally freeze rho near convergence
        if adapt_enabled and gap < cfg.freeze_rho_at * cfg.tol_gap:
            adapt_enabled = False  # stop adapting rho

        # ----- logging
        rec = {
            "iter": it,
            "gap": float(gap),
            "P": float(P),
            "D": float(D),
            "rho": float(rho),
            "r_norm": r_norm,
            "d_norm": d_norm,
            "time": float(time.perf_counter() - t0),
        }
        history.append(rec)
        if cfg.verbose and (it % 10 == 0):
            print(f"[ADMM] it={it:04d} gap={gap:.3e} P={P:.6f} rho={rho:.3g} r={r_norm:.3e} d={d_norm:.3e}")

        if gap <= cfg.tol_gap:
            break

    return w, history

# utils.py
from __future__ import annotations
import numpy as np

try:
    import scipy.sparse as sp
    _HAVE_SP = True
except Exception:
    sp = None
    _HAVE_SP = False


# ---------- numerics ----------
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Stable logistic sigmoid applied elementwise."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def soft_threshold(a: np.ndarray, tau: float | np.ndarray) -> np.ndarray:
    """Soft-thresholding operator S_tau(a)."""
    return np.sign(a) * np.maximum(np.abs(a) - tau, 0.0)


# ---------- primal / dual / gap ----------
def primal_obj(X, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """
    Logistic loss with y in {0,1} plus L1:
        (1/m) sum_i [ log(1+exp(z_i)) - y_i z_i ] + lam * ||w||_1
    Use logaddexp for stable softplus.
    """
    z = X @ w
    loss = np.logaddexp(0.0, z) - y * z
    return float(loss.mean() + lam * np.linalg.norm(w, 1))


def _xt_dot(X, v: np.ndarray) -> np.ndarray:
    """X.T @ v for dense or sparse X."""
    if _HAVE_SP and sp.issparse(X):
        return X.T @ v
    return X.T.dot(v)


def _col_l2_norms(X) -> np.ndarray:
    """Column L2 norms for dense or sparse X."""
    if _HAVE_SP and sp.issparse(X):
        return np.sqrt(np.array(X.power(2).sum(axis=0)).ravel())
    else:
        return np.sqrt((X ** 2).sum(axis=0))


def dual_feasible_from_w(X, y: np.ndarray, w: np.ndarray, lam: float, eps: float = 1e-12):
    """
    Construct a dual-feasible u in [0,1]^m by scaling p towards y so that
        ||X^T(u - y)||_inf <= lam * m.
    Returns: u, theta=(u-y), D(u), alpha, feas_before
    """
    z = X @ w
    p = sigmoid(z)
    s = p - y
    xts = _xt_dot(X, s)
    m = y.shape[0]
    denom = float(np.max(np.abs(xts))) if xts.size else 0.0
    feas_before = denom / (lam * m) if lam > 0 else np.inf

    if denom <= lam * m or denom == 0.0:
        alpha = 1.0
    else:
        alpha = (lam * m) / denom

    u = alpha * p + (1.0 - alpha) * y
    u = np.clip(u, eps, 1.0 - eps)
    theta = u - y

    H = u * np.log(u) + (1 - u) * np.log(1 - u)
    D = - float(H.mean())
    return u, theta, D, alpha, feas_before


def duality_gap(X, y: np.ndarray, w: np.ndarray, lam: float):
    """
    Return: gap, P, D, u, theta
    """
    P = primal_obj(X, y, w, lam)
    u, theta, D, _, _ = dual_feasible_from_w(X, y, w, lam)
    G = P - D
    return float(G), float(P), float(D), u, theta


def gap_safe_radius(m: int, gap: float) -> float:
    """
    Gap-Safe ball radius under our scaling (dual strong concavity μ=4/m):
        r = sqrt( (m/2) * gap )
    """
    return float(np.sqrt(0.5 * m * max(gap, 0.0)))


# ---------- gradients / Hessian-diag for CD ----------
def grad_and_hess_diag(X, y: np.ndarray, w: np.ndarray):
    """
    Return:
      g = (1/m) X^T (p - y)
      h_diag_j = (1/m) sum_i x_{ij}^2 * p_i(1-p_i)
    Works with dense or CSR X.
    """
    m = y.shape[0]
    z = X @ w
    p = sigmoid(z)
    g = _xt_dot(X, (p - y)) / m
    s = p * (1 - p)
    if _HAVE_SP and sp.issparse(X):
        h_diag = (X.power(2).T @ s) / m
        h_diag = np.asarray(h_diag).ravel()
    else:
        h_diag = ((X ** 2).T @ s) / m
    h_diag = np.maximum(h_diag, 1e-12)
    return g, h_diag, p


# ---------- screening ----------
def strong_rule_mask(X, y: np.ndarray, w_prev: np.ndarray, lam_prev: float, lam_new: float) -> np.ndarray:
    """
    Sequential Strong Rule:
      discard j if |c_j_old| < 2*lam_new - lam_prev,
      where c_old = (1/m) X^T (p_prev - y).
    Returns keep_mask (True means keep).
    """
    if lam_new <= 0:
        return np.ones(X.shape[1], dtype=bool)

    m = y.shape[0]
    z_prev = X @ w_prev
    p_prev = sigmoid(z_prev)
    c_old = _xt_dot(X, (p_prev - y)) / m
    thresh = 2 * lam_new - lam_prev
    if thresh <= 0:
        return np.ones_like(c_old, dtype=bool)
    keep = np.abs(c_old) >= thresh
    return keep.astype(bool)


def gap_safe_mask(X, theta: np.ndarray, gap: float, lam: float, col_norms: np.ndarray | None = None) -> np.ndarray:
    """
    Gap-Safe rule with safe ball radius r:
      drop j if |x_j^T theta| + r*||x_j||_2 < lam * m
    Safety guards against non-finite gaps / norms (keep features in doubt).
    """
    m, n = X.shape
    if not np.isfinite(gap) or gap < 0:
        return np.ones(n, dtype=bool)

    r = gap_safe_radius(m, gap)
    if not np.isfinite(r):
        return np.ones(n, dtype=bool)

    if col_norms is None:
        col_norms = _col_l2_norms(X)
    col_norms = np.where(np.isfinite(col_norms), col_norms, 0.0)

    xTtheta = _xt_dot(X, theta)
    scores = np.abs(xTtheta) + r * col_norms
    scores = np.where(np.isfinite(scores), scores, np.inf)

    keep = scores >= (lam * m - 1e-12)
    return keep.astype(bool)


# ---------- diagnostics ----------
def kkt_violations(X, y: np.ndarray, w: np.ndarray, lam: float, active_mask: np.ndarray | None = None):
    """
    Compute per-coordinate KKT residuals:
      r_j = |(1/m) x_j^T (p - y)| - lam
    Returns residual vector and summary stats.
    """
    m = y.shape[0]
    z = X @ w
    p = sigmoid(z)
    c = _xt_dot(X, (p - y)) / m
    res = np.abs(c) - lam
    return res, {
        "max_violation": float(np.maximum(res, 0.0).max(initial=0.0)),
        "mean_violation": float(np.maximum(res, 0.0).mean())
    }

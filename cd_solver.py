# cd_solver.py
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
    duality_gap, strong_rule_mask, gap_safe_mask,
    _col_l2_norms, _xt_dot
)


class CDConfig:
    """
    Coordinate Descent configuration.
    - max_epochs: max outer passes over (current) active coordinates
    - tol_gap: duality-gap stopping threshold
    - gap_safe_every: run Gap-Safe once every this many epochs (0 to disable)
    - recompute_every: refresh p/pv/s after this many coordinate updates (None/0 => disable batch refresh)
    - shuffle: shuffle active coordinate order per epoch
    - use_lipschitz_upper: use L_j = ||x_j||^2/(4m) as an upper curvature (safer step)
    - kkt_every: run KKT-repair once every this many epochs (>=1)
    - verbose: print per-epoch logs
    """
    def __init__(
        self,
        max_epochs: int = 50,
        tol_gap: float = 1e-4,
        gap_safe_every: int = 1,
        verbose: bool = False,
        recompute_every: int | None = 128,
        shuffle: bool = True,
        use_lipschitz_upper: bool = True,
        kkt_every: int = 1,
    ):
        self.max_epochs = max_epochs
        self.tol_gap = tol_gap
        self.gap_safe_every = gap_safe_every
        self.verbose = verbose
        self.recompute_every = recompute_every
        self.shuffle = shuffle
        self.use_lipschitz_upper = use_lipschitz_upper
        self.kkt_every = max(1, kkt_every)


def _get_column_sparse_view(X_csc, j: int):
    """Return (rows, data) for column j in CSC without densifying."""
    start, end = X_csc.indptr[j], X_csc.indptr[j + 1]
    rows = X_csc.indices[start:end]
    data = X_csc.data[start:end]
    return rows, data


def _get_column_dense(X, j: int):
    """Dense column view as 1D array."""
    return X[:, j]


def cd_solve_lambda(
    X, y: np.ndarray, lam: float,
    w0: np.ndarray | None = None,
    strong_keep: np.ndarray | None = None,
    cfg: CDConfig = CDConfig(),
    precomputed_col_norms: np.ndarray | None = None,
):
    """
    Coordinate Descent for a single lambda with:
      - optional Strong Rule initial screening (strong_keep mask)
      - periodic Gap-Safe screening
      - duality-gap stopping

    Returns:
      w           : solution vector
      history     : list of dicts (epoch logs)
      active_mask : final active mask (True = keep)
    """
    t0 = time.perf_counter()
    m, n = X.shape
    sparse = _HAVE_SP and sp.issparse(X)

    # Prepare CSR for fast matvec; CSC for fast column access
    X_csr = X.tocsr() if (sparse and not sp.isspmatrix_csr(X)) else (X if sparse else None)
    X_csc = X.tocsc() if sparse else None

    w = np.zeros(n) if w0 is None else w0.copy()
    active = np.ones(n, dtype=bool)

    # Track which dropped features are allowed to be revived by KKT
    # Only those dropped heuristically by Strong Rule are revivable.
    revivable = np.zeros(n, dtype=bool)

    if strong_keep is not None:
        active &= strong_keep
        revivable[~strong_keep] = True   # Strong-dropped can be revived
        w[~active] = 0.0

    # working response z = Xw
    z = (X @ w) if not sparse else (X_csr @ w if X_csr is not None else X @ w)

    # precompute col norms once if not provided
    col_norms = precomputed_col_norms
    if col_norms is None:
        col_norms = _col_l2_norms(X)

    # ---- NEW: Lipschitz upper bounds L_j = ||x_j||^2 / (4m) to stabilize steps
    lipschitz = (col_norms ** 2) / (4.0 * m)
    lipschitz = np.maximum(lipschitz, 1e-12)

    history = []
    for epoch in range(cfg.max_epochs):
        # --- epoch start: refresh once
        p = sigmoid(z)
        pv = (p - y)
        s = p * (1 - p)

        active_idx = np.where(active)[0]
        if cfg.shuffle:
            np.random.shuffle(active_idx)

        updates = 0
        for j in active_idx:
            if sparse:
                rows, vals = _get_column_sparse_view(X_csc, j)
                g_j = float((vals * pv[rows]).sum()) / m
                h_j = float((vals * vals * s[rows]).sum()) / m + 1e-12
            else:
                xj = _get_column_dense(X, j)
                g_j = float(xj.dot(pv)) / m
                h_j = float((xj * xj).dot(s)) / m + 1e-12

            # ---- stable denom: use Lipschitz upper bound (>= h_j) for safe MM step
            if cfg.use_lipschitz_upper:
                denom = max(h_j, lipschitz[j])  # == lipschitz[j] in practice
            else:
                denom = h_j

            # soft-thresholded proximal step with 'denom'
            w_old = w[j]
            w_new = soft_threshold(w_old - g_j / denom, lam / denom)
            delta = w_new - w_old

            if delta != 0.0:
                w[j] = w_new
                if sparse:
                    z[rows] += delta * vals
                else:
                    z += delta * xj

            updates += 1
            # batch refresh: keep p/pv/s reasonably fresh
            if cfg.recompute_every and cfg.recompute_every > 0 and (updates % cfg.recompute_every == 0):
                p = sigmoid(z)
                pv = (p - y)
                s = p * (1 - p)

        # --- ensure freshest p/pv/s before certification or KKT
        p = sigmoid(z)
        pv = (p - y)
        s = p * (1 - p)

        # --- certification (duality gap) & logging
        gap, P, D, u, theta = duality_gap(X, y, w, lam)
        rec = {
            "epoch": epoch,
            "gap": float(gap),
            "P": float(P),
            "D": float(D),
            "active": int(active.sum()),
            "time": float(time.perf_counter() - t0),
        }
        history.append(rec)
        if cfg.verbose:
            print(f"[CD] ep={epoch:02d} gap={gap:.3e} P={P:.6f} active={active.sum()} time={rec['time']:.2f}s")

        # Stop if certified
        if gap <= cfg.tol_gap:
            break

        # --- KKT re-activation (repair Strong mistakes only), optionally sub-sampled by epochs
        if (epoch % cfg.kkt_every) == 0:
            # Re-activate only if |(1/m) x_j^T (p - y)| > lam + tol  AND the feature was Strong-dropped.
            c = _xt_dot(X, pv) / m
            tol_reactivate = 1e-8
            violated = (~active) & revivable & (np.abs(c) > (lam + tol_reactivate))
            if np.any(violated):
                active[violated] = True
                if cfg.verbose:
                    print(f"[CD] reactivated {int(violated.sum())} features by KKT check")

        # --- Gap-Safe screening (periodic, after KKT repair)
        if cfg.gap_safe_every and (epoch % cfg.gap_safe_every == 0):
            keep_mask = gap_safe_mask(X, theta, gap, lam, col_norms=col_norms)
            newly_safe_dropped = active & (~keep_mask)
            dropped = int(newly_safe_dropped.sum())
            if dropped > 0:
                # safe-dropped are NOT revivable anymore
                revivable[newly_safe_dropped] = False
                active &= keep_mask
                w[~active] = 0.0
                if cfg.verbose:
                    print(f"[CD] gap-safe dropped {dropped} features; active={active.sum()}")

    return w, history, active


def cd_path(
    X, y: np.ndarray, lambdas: np.ndarray,
    cfg: CDConfig = CDConfig(),
    warm_start: bool = True,
):
    """
    Solve a descending lambda path with Strong->Gap-Safe screening per lambda.
    Returns:
      W_list   : list of solutions per lambda
      logs     : list of histories per lambda
    """
    assert lambdas.ndim == 1, "lambdas must be 1-D array"
    diffs = np.diff(lambdas)
    assert np.all(diffs <= 1e-12), "Pass lambdas in descending order."

    m, n = X.shape
    W_list, logs = [], []
    w_prev = np.zeros(n)
    col_norms = _col_l2_norms(X)

    lam_prev = None
    for k, lam in enumerate(lambdas):
        # Strong Rule pre-screen
        strong_keep = None
        if k > 0 and lam_prev is not None:
            strong_keep = strong_rule_mask(X, y, w_prev, lam_prev, lam)

        w0 = w_prev if warm_start else np.zeros(n)
        w_k, hist_k, active_k = cd_solve_lambda(
            X, y, lam,
            w0=w0,
            strong_keep=strong_keep,
            cfg=cfg,
            precomputed_col_norms=col_norms
        )
        W_list.append(w_k)
        logs.append({
            "lambda": float(lam),
            "history": hist_k,
            "active_final": int(active_k.sum()),
        })

        w_prev = w_k
        lam_prev = lam

    return W_list, logs

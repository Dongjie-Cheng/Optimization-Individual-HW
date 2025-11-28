# experiment.py
from __future__ import annotations
import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

try:
    import scipy.sparse as sp
    _HAVE_SP = True
except Exception:
    sp = None
    _HAVE_SP = False

from data import (
    load_20ng_binary, load_rcv1_binary,
    train_val_test_split_sparse, ensure_csr
)
from datasets_v2 import (
    load_a9a_binary, load_ijcnn1_binary, load_realsim_binary
)
from utils import duality_gap, sigmoid, _xt_dot
from cd_solver import CDConfig, cd_path
from admm_solver import ADMMConfig, admm_solve


# -----------------------------
# Lambda utilities
# -----------------------------
def compute_lambda_max(X, y) -> float:
    """
    For logistic (y in {0,1}), with
        P(w) = (1/m) sum log(1+exp(z_i)) - y_i z_i + lam||w||_1,
    KKT at w=0 gives: lambda_max = (1/m) * || X^T (y - 0.5) ||_inf
    """
    m = y.shape[0]
    c = _xt_dot(X, (y - 0.5))
    lam_max = float(np.max(np.abs(c)) / m)
    return lam_max


def make_lambda_path(lam_max: float, gamma: float = 1e-2, K: int = 50) -> np.ndarray:
    """
    Geometric path from lam_max down to lam_max*gamma (descending).
    """
    ks = np.arange(K, dtype=float)
    path = lam_max * (gamma ** (ks / (K - 1)))
    return path  # descending by construction (k increases => lam decreases)


def pick_regime_indices(lams: np.ndarray, lam_max: float | None = None) -> Dict[str, int]:
    """
    Pick indices for sparse/mid/dense regimes by position along the path
    rather than target lambda values.

    We take roughly:
      sparse ~ 10% of the path (close to lambda_max, very sparse solution)
      mid    ~ 50% of the path
      dense  ~ 90% of the path (close to lambda_min, dense solution)
    """
    K = len(lams)
    if K < 3:
        # degenerate case: fall back to start/middle/end best-effort
        return {"sparse": 0, "mid": K // 2, "dense": K - 1}

    def pos(frac: float) -> int:
        # map frac in [0,1] to index [0, K-1]
        return int(round(frac * (K - 1)))

    idx = {
        "sparse": pos(0.1),
        "mid":    pos(0.5),
        "dense":  pos(0.9),
    }
    return idx


# -----------------------------
# Evaluation helpers
# -----------------------------
def time_to_epsilon_from_cd_history(history: List[Dict[str, Any]], eps: float) -> float:
    """
    CD logs are per-epoch; use the first time with gap <= eps.
    If not reached, return final time (acts as ceiling).
    """
    if not history:
        return math.inf
    times = [h["time"] for h in history]
    gaps = [h["gap"] for h in history]
    for t, g in zip(times, gaps):
        if g <= eps:
            return float(t)
    return float(times[-1])


def time_to_epsilon_from_admm_history(history: List[Dict[str, Any]], eps: float) -> float:
    """
    ADMM logs are per-iter; use the first time with gap <= eps.
    """
    if not history:
        return math.inf
    times = [h["time"] for h in history]
    gaps = [h["gap"] for h in history]
    for t, g in zip(times, gaps):
        if g <= eps:
            return float(t)
    return float(times[-1])


# -----------------------------
# Experiment driver
# -----------------------------
@dataclass
class RunConfig:
    # dataset: one of {"20ng", "rcv1", "a9a", "ijcnn1", "real-sim"}
    dataset: str = "20ng"

    # for 20NG
    vocab_size: int = 50_000
    min_df: int = 5

    # for RCV1
    rcv1_label: str = "CCAT"

    # for LIBSVM datasets (a9a, ijcnn1, real-sim)
    data_root: str = "data/libsvm"  # folder containing a9a, a9a.t, ijcnn1, ijcnn1.t, real-sim

    seed: int = 0
    gamma: float = 1e-2            # path end ratio
    K: int = 50                    # path length
    eps: float = 1e-4              # duality gap tolerance

    cd_max_epochs: int = 50
    admm_max_iters: int = 2000
    admm_alpha: float = 1.5
    admm_rho_init: float = 1.0
    admm_newton_steps: int = 1

    verbose: bool = True
    save_json: str | None = None


def load_dataset(name: str, cfg: RunConfig):
    """
    Returns:
      Xtr, ytr, Xval, yval, Xte, yte  (all as CSR, y as float {0,1})
    """
    name = name.lower()
    if name == "20ng":
        X, y, _ = load_20ng_binary(
            vocab_size=cfg.vocab_size,
            min_df=cfg.min_df,
            random_state=cfg.seed
        )
    elif name == "rcv1":
        X, y, _ = load_rcv1_binary(label=cfg.rcv1_label, subset="all")
    elif name == "a9a":
        X, y = load_a9a_binary(cfg.data_root)
    elif name == "ijcnn1":
        X, y = load_ijcnn1_binary(cfg.data_root)
    elif name in ("real-sim", "realsim"):
        X, y = load_realsim_binary(cfg.data_root)
    else:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Expected one of: 20ng, rcv1, a9a, ijcnn1, real-sim.")

    (Xtr, ytr), (Xval, yval), (Xte, yte) = train_val_test_split_sparse(X, y, seed=cfg.seed)
    return ensure_csr(Xtr), ytr, ensure_csr(Xval), yval, ensure_csr(Xte), yte


def run_experiment(cfg: RunConfig) -> Dict[str, Any]:
    # ----- data
    Xtr, ytr, Xval, yval, Xte, yte = load_dataset(cfg.dataset, cfg)

    # ----- path
    lam_max = compute_lambda_max(Xtr, ytr)
    lams = make_lambda_path(lam_max, gamma=cfg.gamma, K=cfg.K)
    reg_idx = pick_regime_indices(lams, lam_max)

    if cfg.verbose:
        print(f"[INFO] dataset={cfg.dataset}, m={Xtr.shape[0]}, n={Xtr.shape[1]}")
        print(f"[INFO] lambda_max={lam_max:.6e}, path[{cfg.K}] from {lams[0]:.3e} to {lams[-1]:.3e}")
        print(f"[INFO] regimes indices: {reg_idx} -> values: "
              f"{ {k: float(lams[i]) for k,i in reg_idx.items()} }")

    # ----- CD over full path (with screening)
    cd_cfg = CDConfig(
        max_epochs=cfg.cd_max_epochs,
        tol_gap=cfg.eps,
        gap_safe_every=1,
        verbose=cfg.verbose,
        recompute_every=128,         # slightly safer default
        # use_lipschitz_upper=True,  # if your CDConfig has this arg, keep it True
    )
    t_cd0 = time.perf_counter()
    W_list, logs_cd = cd_path(Xtr, ytr, lams, cfg=cd_cfg, warm_start=True)
    cd_time_total = float(time.perf_counter() - t_cd0)

    # compute time-to-eps for the three regimes
    t2eps_cd = {}
    for k, i in reg_idx.items():
        hist = logs_cd[i]["history"]
        t2eps_cd[k] = time_to_epsilon_from_cd_history(hist, cfg.eps)

    # ----- ADMM at the three regimes
    admm_cfg = ADMMConfig(
        max_iters=cfg.admm_max_iters,
        tol_gap=cfg.eps,
        rho_init=cfg.admm_rho_init,
        alpha=cfg.admm_alpha,
        newton_steps=cfg.admm_newton_steps,
        verbose=cfg.verbose,
    )
    n = Xtr.shape[1]

    admm_runs = {}
    for k, i in reg_idx.items():
        lam = float(lams[i])
        if cfg.verbose:
            print(f"[ADMM] run at regime={k} (lambda={lam:.3e})")
        # 从零向量开始，作为独立 baseline（与 CD 一样从头优化）
        w0 = np.zeros(n, dtype=float)
        w_admm, hist_admm = admm_solve(Xtr, ytr, lam, w0=w0, cfg=admm_cfg)
        admm_runs[k] = {
            "lambda": lam,
            "time_to_eps": time_to_epsilon_from_admm_history(hist_admm, cfg.eps),
            "history": hist_admm,
        }

    # ----- summary
    summary = {
        "config": asdict(cfg),
        "lambda_max": lam_max,
        "lambdas": lams.tolist(),
        "regimes": {k: int(i) for k, i in reg_idx.items()},
        "cd": {
            "total_time": cd_time_total,
            "time_to_eps": t2eps_cd,
            "logs": logs_cd,  # per-lambda histories
        },
        "admm": admm_runs,
        # 注意：Xval/Xte 和泛化指标你可以在这里再加，比如 test accuracy / nnz 等
    }

    if cfg.save_json:
        with open(cfg.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f)
        if cfg.verbose:
            print(f"[SAVE] results saved to {cfg.save_json}")

    return summary


def build_argparser():
    p = argparse.ArgumentParser(description="Gap-certified L1-logistic: CD path + ADMM regimes")
    p.add_argument("--dataset", type=str, default="20ng",
                   choices=["20ng", "rcv1", "a9a", "ijcnn1", "real-sim"])

    # 20NG
    p.add_argument("--vocab_size", type=int, default=50_000)
    p.add_argument("--min_df", type=int, default=5)

    # RCV1
    p.add_argument("--rcv1_label", type=str, default="CCAT")

    # LIBSVM
    p.add_argument("--data_root", type=str, default="data/libsvm",
                   help="Folder containing a9a, a9a.t, ijcnn1, ijcnn1.t, real-sim")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=1e-2)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-4)

    p.add_argument("--cd_max_epochs", type=int, default=50)

    p.add_argument("--admm_max_iters", type=int, default=2000)
    p.add_argument("--admm_alpha", type=float, default=1.5)
    p.add_argument("--admm_rho_init", type=float, default=1.0)
    p.add_argument("--admm_newton_steps", type=int, default=1)

    p.add_argument("--save_json", type=str, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = RunConfig(
        dataset=args.dataset,
        vocab_size=args.vocab_size,
        min_df=args.min_df,
        rcv1_label=args.rcv1_label,
        data_root=args.data_root,
        seed=args.seed,
        gamma=args.gamma,
        K=args.K,
        eps=args.eps,
        cd_max_epochs=args.cd_max_epochs,
        admm_max_iters=args.admm_max_iters,
        admm_alpha=args.admm_alpha,
        admm_rho_init=args.admm_rho_init,
        admm_newton_steps=args.admm_newton_steps,
        verbose=(not args.quiet),
        save_json=args.save_json,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()

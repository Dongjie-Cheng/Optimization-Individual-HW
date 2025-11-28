# ablation.py
from __future__ import annotations
import argparse
import csv
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

import numpy as np

try:
    import scipy.sparse as sp
    _HAVE_SP = True
except Exception:
    sp = None
    _HAVE_SP = False

from data import load_20ng_binary, load_rcv1_binary, train_val_test_split_sparse, ensure_csr
from utils import duality_gap, strong_rule_mask, gap_safe_mask, _xt_dot, gap_safe_radius, _col_l2_norms, sigmoid
from cd_solver import CDConfig, cd_solve_lambda
from admm_solver import ADMMConfig, admm_solve
from experiment import compute_lambda_max, make_lambda_path, pick_regime_indices, time_to_epsilon_from_cd_history, time_to_epsilon_from_admm_history


# -----------------------------
# Configs
# -----------------------------
@dataclass
class AblConfig:
    dataset: str = "20ng"       # "20ng" or "rcv1"
    vocab_size: int = 50_000
    min_df: int = 5
    rcv1_label: str = "CCAT"

    seeds: str = "0,1,2"        # comma-separated
    eps_list: str = "1e-3,1e-4" # comma-separated
    gamma: float = 1e-2         # path end ratio
    K: int = 50                 # path length

    # CD knobs
    cd_max_epochs: int = 50
    # ADMM knobs (defaults; grid set below)
    admm_max_iters: int = 2000
    admm_newton_steps: int = 1

    out_csv: str = "ablation_results.csv"
    verbose: bool = True


# -----------------------------
# Dataset loader
# -----------------------------
def load_dataset(name: str, cfg: AblConfig):
    if name == "20ng":
        X, y, _ = load_20ng_binary(vocab_size=cfg.vocab_size, min_df=cfg.min_df)
    elif name == "rcv1":
        X, y, _ = load_rcv1_binary(label=cfg.rcv1_label, subset="all")
    else:
        raise ValueError("dataset must be '20ng' or 'rcv1'")
    return X, y


def split_dataset(X, y, seed: int):
    (Xtr, ytr), (Xval, yval), (Xte, yte) = train_val_test_split_sparse(X, y, seed=seed)
    return ensure_csr(Xtr), ytr, ensure_csr(Xval), yval, ensure_csr(Xte), yte


# -----------------------------
# CD runners (screen vs no-screen)
# -----------------------------
def run_cd_path(
    Xtr, ytr, lams: np.ndarray, eps: float, screening: bool, max_epochs: int, verbose: bool
):
    """
    Run CD along path; if screening=False => no Strong, no Gap-Safe.
    Returns per-lambda histories and final w per lambda.
    """
    m, n = Xtr.shape
    w_prev = np.zeros(n)
    histories: List[List[Dict[str, Any]]] = []
    W_list: List[np.ndarray] = []
    lam_prev = None
    col_norms = _col_l2_norms(Xtr)
    cfg = CDConfig(max_epochs=max_epochs, tol_gap=eps, gap_safe_every=(1 if screening else 0), verbose=verbose)

    for k, lam in enumerate(lams):
        strong_keep = None
        if screening and (k > 0) and (lam_prev is not None):
            strong_keep = strong_rule_mask(Xtr, ytr, w_prev, lam_prev, lam)

        w_k, hist_k, active_k = cd_solve_lambda(
            Xtr, ytr, lam,
            w0=w_prev,
            strong_keep=(strong_keep if screening else None),
            cfg=cfg,
            precomputed_col_norms=col_norms
        )
        histories.append(hist_k)
        W_list.append(w_k)
        w_prev = w_k
        lam_prev = lam

    return W_list, histories


# -----------------------------
# ADMM runners (fixed rho grid / adaptive with alpha grid)
# -----------------------------
def run_admm_single(
    Xtr, ytr, lam: float, eps: float,
    rho_init: float | None, alpha: float,
    max_iters: int, newton_steps: int, verbose: bool
):
    cfg = ADMMConfig(
        max_iters=max_iters,
        tol_gap=eps,
        rho_init=(rho_init if rho_init is not None else 1.0),
        alpha=alpha,
        newton_steps=newton_steps,
        verbose=verbose,
    )
    # If rho_init is None, we interpret as "adaptive" (ADMMConfig already adapts).
    # If we want to disable adaptation for fixed-rho, set adapt_every=0
    if rho_init is not None:
        cfg.adapt_every = 0  # fixed rho
    w, hist = admm_solve(Xtr, ytr, lam, w0=None, cfg=cfg)
    return w, hist


# -----------------------------
# Main ablation routine
# -----------------------------
def ablation(cfg: AblConfig):
    # parse lists
    seed_list = [int(s) for s in cfg.seeds.split(",") if s.strip() != ""]
    eps_list = [float(eval(s)) for s in cfg.eps_list.split(",")]  # allow 1e-4 literal

    # open CSV
    fieldnames = [
        "dataset", "seed", "eps", "lambda_idx", "lambda_value",
        "method", "screening", "rho", "alpha",
        "time_to_eps", "final_gap", "iters_or_epochs", "nnz"
    ]
    with open(cfg.out_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        # load dataset once (full), split per seed
        X_all, y_all = load_dataset(cfg.dataset, cfg)

        for seed in seed_list:
            if cfg.verbose:
                print(f"\n===== Seed {seed} =====")
            Xtr, ytr, Xval, yval, Xte, yte = split_dataset(X_all, y_all, seed=seed)

            lam_max = compute_lambda_max(Xtr, ytr)
            lams = make_lambda_path(lam_max, gamma=cfg.gamma, K=cfg.K)
            reg_idx = pick_regime_indices(lams, lam_max)

            if cfg.verbose:
                print(f"[INFO] m={Xtr.shape[0]} n={Xtr.shape[1]}  lambda_max={lam_max:.3e}")
                print(f"[INFO] regimes: { {k: float(lams[i]) for k,i in reg_idx.items()} }")

            # ---- CD: screen vs no-screen along full path (collect regimes)
            for eps in eps_list:
                # with screening
                W_scr, H_scr = run_cd_path(Xtr, ytr, lams, eps=eps, screening=True, max_epochs=cfg.cd_max_epochs, verbose=False)
                # without screening
                W_nos, H_nos = run_cd_path(Xtr, ytr, lams, eps=eps, screening=False, max_epochs=cfg.cd_max_epochs, verbose=False)

                for regime, idx in reg_idx.items():
                    lam = float(lams[idx])

                    # CD screen
                    t2e = time_to_epsilon_from_cd_history(H_scr[idx], eps)
                    final_gap = H_scr[idx][-1]["gap"] if H_scr[idx] else math.inf
                    nnz = int(np.count_nonzero(W_scr[idx]))
                    writer.writerow({
                        "dataset": cfg.dataset, "seed": seed, "eps": eps,
                        "lambda_idx": idx, "lambda_value": lam,
                        "method": "CD", "screening": 1, "rho": "", "alpha": "",
                        "time_to_eps": t2e, "final_gap": final_gap,
                        "iters_or_epochs": len(H_scr[idx]), "nnz": nnz
                    })

                    # CD no-screen
                    t2e = time_to_epsilon_from_cd_history(H_nos[idx], eps)
                    final_gap = H_nos[idx][-1]["gap"] if H_nos[idx] else math.inf
                    nnz = int(np.count_nonzero(W_nos[idx]))
                    writer.writerow({
                        "dataset": cfg.dataset, "seed": seed, "eps": eps,
                        "lambda_idx": idx, "lambda_value": lam,
                        "method": "CD", "screening": 0, "rho": "", "alpha": "",
                        "time_to_eps": t2e, "final_gap": final_gap,
                        "iters_or_epochs": len(H_nos[idx]), "nnz": nnz
                    })

                # ---- ADMM: fixed-rho grid and adaptive with alpha grid, per regime
                fixed_rhos = [0.1, 1.0, 10.0]
                alphas = [1.0, 1.5, 1.8]

                for regime, idx in reg_idx.items():
                    lam = float(lams[idx])

                    # fixed-rho (alpha=1.0 only to keep grid small)
                    for rho in fixed_rhos:
                        w, hist = run_admm_single(
                            Xtr, ytr, lam, eps=eps,
                            rho_init=rho, alpha=1.0,
                            max_iters=cfg.admm_max_iters, newton_steps=cfg.admm_newton_steps, verbose=False
                        )
                        t2e = time_to_epsilon_from_admm_history(hist, eps)
                        final_gap = hist[-1]["gap"] if hist else math.inf
                        nnz = int(np.count_nonzero(w))
                        writer.writerow({
                            "dataset": cfg.dataset, "seed": seed, "eps": eps,
                            "lambda_idx": idx, "lambda_value": lam,
                            "method": "ADMM", "screening": "", "rho": rho, "alpha": 1.0,
                            "time_to_eps": t2e, "final_gap": final_gap,
                            "iters_or_epochs": len(hist), "nnz": nnz
                        })

                    # adaptive (rho adaptive; test alpha grid)
                    for alpha in alphas:
                        w, hist = run_admm_single(
                            Xtr, ytr, lam, eps=eps,
                            rho_init=None, alpha=alpha,
                            max_iters=cfg.admm_max_iters, newton_steps=cfg.admm_newton_steps, verbose=False
                        )
                        t2e = time_to_epsilon_from_admm_history(hist, eps)
                        final_gap = hist[-1]["gap"] if hist else math.inf
                        nnz = int(np.count_nonzero(w))
                        writer.writerow({
                            "dataset": cfg.dataset, "seed": seed, "eps": eps,
                            "lambda_idx": idx, "lambda_value": lam,
                            "method": "ADMM(adaptive)", "screening": "", "rho": "", "alpha": alpha,
                            "time_to_eps": t2e, "final_gap": final_gap,
                            "iters_or_epochs": len(hist), "nnz": nnz
                        })

    if cfg.verbose:
        print(f"[SAVE] ablation results saved -> {cfg.out_csv}")


# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Ablation runner for gap-certified L1-logistic")
    p.add_argument("--dataset", type=str, default="20ng", choices=["20ng", "rcv1"])
    p.add_argument("--vocab_size", type=int, default=50_000)
    p.add_argument("--min_df", type=int, default=5)
    p.add_argument("--rcv1_label", type=str, default="CCAT")

    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--eps_list", type=str, default="1e-3,1e-4")
    p.add_argument("--gamma", type=float, default=1e-2)
    p.add_argument("--K", type=int, default=50)

    p.add_argument("--cd_max_epochs", type=int, default=50)
    p.add_argument("--admm_max_iters", type=int, default=2000)
    p.add_argument("--admm_newton_steps", type=int, default=1)

    p.add_argument("--out_csv", type=str, default="ablation_results.csv")
    p.add_argument("--quiet", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = AblConfig(
        dataset=args.dataset,
        vocab_size=args.vocab_size,
        min_df=args.min_df,
        rcv1_label=args.rcv1_label,
        seeds=args.seeds,
        eps_list=args.eps_list,
        gamma=args.gamma,
        K=args.K,
        cd_max_epochs=args.cd_max_epochs,
        admm_max_iters=args.admm_max_iters,
        admm_newton_steps=args.admm_newton_steps,
        out_csv=args.out_csv,
        verbose=(not args.quiet),
    )
    ablation(cfg)


if __name__ == "__main__":
    main()

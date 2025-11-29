# plots.py
from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# I/O
# -----------------------------
def load_summary(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_outdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------
# Helpers
# -----------------------------
def _pick_regimes(summary: Dict[str, Any]) -> Dict[str, int]:
    return {k: int(i) for k, i in summary["regimes"].items()}


def _cd_history_at(summary: Dict[str, Any], idx: int) -> List[Dict[str, Any]]:
    return summary["cd"]["logs"][idx]["history"]


def _admm_history_at(summary: Dict[str, Any], regime: str) -> List[Dict[str, Any]]:
    return summary["admm"][regime]["history"]


def _dataset_tag(summary: Dict[str, Any]) -> str:
    """Short dataset tag used in titles / filenames."""
    cfg = summary.get("config", {})
    ds = cfg.get("dataset", "dataset")
    # safer for filenames
    return str(ds).replace(" ", "_").replace("/", "-")


# -----------------------------
# Plots
# -----------------------------
def plot_cd_gap_vs_time(
    summary: Dict[str, Any],
    indices: List[int],
    outdir: Path,
    title_suffix: str = "",
):
    """
    For selected lambda indices, plot CD gap vs time curves (log-y).
    """
    ds = _dataset_tag(summary)
    plt.figure()
    for i in indices:
        hist = _cd_history_at(summary, i)
        t = [h["time"] for h in hist]
        g = [h["gap"] for h in hist]
        lam = summary["lambdas"][i]
        plt.plot(t, g, label=f"CD λ[{i}]={lam:.2e}")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Duality gap")
    plt.title(f"CD Gap vs Time on {ds} {title_suffix}".strip())
    plt.legend()
    idx_str = "-".join(map(str, indices))
    out = outdir / f"cd_gap_time_{ds}_idx-{idx_str}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_cd_active_vs_time(
    summary: Dict[str, Any],
    idx: int,
    outdir: Path,
    title_suffix: str = "",
):
    """
    Plot active feature count vs time for one lambda index (CD).
    """
    ds = _dataset_tag(summary)
    hist = _cd_history_at(summary, idx)
    t = [h["time"] for h in hist]
    a = [h["active"] for h in hist]
    lam = summary["lambdas"][idx]

    plt.figure()
    plt.plot(t, a, marker="o")
    plt.xlabel("Time (s)")
    plt.ylabel("Active features")
    plt.title(
        f"CD Active Set vs Time on {ds} (λ[{idx}]={lam:.2e}) {title_suffix}".strip()
    )
    out = outdir / f"cd_active_time_{ds}_idx-{idx}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_admm_gap_vs_time(
    summary: Dict[str, Any],
    regimes: List[str],
    outdir: Path,
    title_suffix: str = "",
):
    """
    Plot ADMM gap vs time for selected regimes (sparse/mid/dense).
    """
    ds = _dataset_tag(summary)
    plt.figure()
    for r in regimes:
        hist = _admm_history_at(summary, r)
        t = [h["time"] for h in hist]
        g = [h["gap"] for h in hist]
        lam = summary["admm"][r]["lambda"]
        plt.plot(t, g, label=f"ADMM {r} λ={lam:.2e}")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Duality gap")
    plt.title(f"ADMM Gap vs Time on {ds} {title_suffix}".strip())
    plt.legend()
    reg_str = "-".join(regimes)
    out = outdir / f"admm_gap_time_{ds}_{reg_str}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_time_to_epsilon(
    summary: Dict[str, Any],
    outdir: Path,
    title_suffix: str = "",
):
    """
    Bar chart: time-to-ε for CD and ADMM across regimes.
    Uses summary['cd']['time_to_eps'] and summary['admm'][reg]['time_to_eps'].
    """
    ds = _dataset_tag(summary)
    regimes = ["sparse", "mid", "dense"]
    cd_t = [summary["cd"]["time_to_eps"][r] for r in regimes]
    admm_t = [summary["admm"][r]["time_to_eps"] for r in regimes]

    x = np.arange(len(regimes))
    w = 0.35

    plt.figure()
    plt.bar(x - w / 2, cd_t, width=w, label="CD")
    plt.bar(x + w / 2, admm_t, width=w, label="ADMM")

    plt.xticks(x, regimes)
    plt.ylabel(f"Time to ε (ε={summary['config']['eps']}) [s]")
    plt.title(f"Time-to-ε on {ds} {title_suffix}".strip())
    plt.legend()
    out = outdir / f"time_to_eps_{ds}_cd-vs-admm.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="Plot utilities for gap-certified L1-logistic experiments"
    )
    p.add_argument(
        "--summary", type=str, required=True, help="Path to JSON produced by experiment.py"
    )
    p.add_argument(
        "--outdir", type=str, default="figs", help="Directory to save figures"
    )
    p.add_argument(
        "--cd_indices",
        type=str,
        default="",
        help="Comma-separated lambda indices to plot for CD",
    )
    p.add_argument(
        "--title_suffix",
        type=str,
        default="",
        help="Extra text for plot titles (e.g. 'eps=1e-4')",
    )
    return p


def main():
    args = build_argparser().parse_args()
    summary = load_summary(args.summary)
    outdir = ensure_outdir(args.outdir)
    title_suffix = args.title_suffix

    regimes = _pick_regimes(summary)

    # CD: gap vs time (user-chosen indices or regimes)
    if args.cd_indices.strip():
        cd_indices = [int(s) for s in args.cd_indices.split(",")]
    else:
        cd_indices = list(regimes.values())
    out1 = plot_cd_gap_vs_time(summary, cd_indices, outdir, title_suffix)

    # CD: active vs time (plot for 'mid' regime by default)
    mid_idx = regimes.get("mid", cd_indices[0])
    out2 = plot_cd_active_vs_time(summary, mid_idx, outdir, title_suffix)

    # ADMM: gap vs time
    out3 = plot_admm_gap_vs_time(summary, ["sparse", "mid", "dense"], outdir, title_suffix)

    # Time-to-epsilon bars
    out4 = plot_time_to_epsilon(summary, outdir, title_suffix)

    print("[SAVED]")
    print(out1)
    print(out2)
    print(out3)
    print(out4)


if __name__ == "__main__":
    main()

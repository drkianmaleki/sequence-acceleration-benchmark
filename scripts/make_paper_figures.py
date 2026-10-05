"""Optional summary figures from the committed results (the paper has no figures).

Run from the repository root:
    python scripts/make_paper_figures.py [--results DIR] [--out DIR]

Outputs fig1_accuracy_vs_catastrophe.pdf (median error vs catastrophe rate of
every non-oracle method at the headline stratum, the excluded set marked) and
fig2_real_curves.pdf (two recorded real curves with the legacy 18-cell
predictions) into paper_figures/ (git-ignored) or --out.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import src.config as C                      # noqa: E402
from src.dangerous import load_dangerous    # noqa: E402

plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "pdf.fonttype": 42})


def parse_args():
    p = argparse.ArgumentParser(description="Optional summary figures (redesign v2)")
    p.add_argument("--results", default=os.path.join(_ROOT, "results"))
    p.add_argument("--out", default=os.path.join(_ROOT, "paper_figures"))
    return p.parse_args()


def fig1(res, out):
    g = pd.read_csv(os.path.join(res, "phase1", "phase1_global.csv"))
    g = g[(g.target_g == C.HEADLINE_G) & (g.is_oracle == 0)]
    # The excluded set comes from the Phase-1 artifact of the same results tree
    # (pooled valid_rate below RANK_MIN_VALID); an absent artifact marks nothing.
    dang = set(load_dangerous(os.path.join(res, "phase1", os.path.basename(C.DANGEROUS_ARTIFACT)),
                              required=False))
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    for _, r in g.iterrows():
        if not np.isfinite(r.med_error):
            continue
        d = r.method in dang
        ax.scatter(r.med_error, 100 * r.cat_rate, s=26 if d else 20,
                   marker="x" if d else "o",
                   color="#c62828" if d else "#1565c0", alpha=0.85,
                   linewidths=1.2, zorder=3)
    for m in ["rational_fit", "richardson_1", "last_value",
              "richardson_a20", "levin_t2", "linear", "log_linear", "pade_22"]:
        s = g[g.method == m]
        if len(s) and np.isfinite(s.iloc[0].med_error):
            r = s.iloc[0]
            ax.annotate(m, (r.med_error, 100 * r.cat_rate),
                        textcoords="offset points", xytext=(8, 4), fontsize=7.2)
    ax.set_xscale("log")
    ax.set_xlabel(f"Median absolute error at g = {C.HEADLINE_G:g}, valid records (log scale)")
    ax.set_ylabel("Catastrophe rate (%)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", ls="", color="#1565c0", label="method"),
        Line2D([0], [0], marker="x", ls="", color="#c62828",
               label=f"excluded (valid rate below {C.RANK_MIN_VALID:g})")],
        loc="upper left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_accuracy_vs_catastrophe.pdf"))
    plt.close(fig)


def fig2(res, out):
    cur = pd.read_csv(os.path.join(res, "real_data", "real_data_curves.csv"))
    rd = pd.read_csv(os.path.join(res, "real_data", "real_data_results.csv"))
    mcol = {"richardson_1": "#e65100", "rational_fit": "#2e7d32"}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharex=True)
    for ax, ds in zip(axes, ["adult", "higgs"]):
        y = cur[ds].to_numpy(); x = np.arange(1, len(y) + 1)
        ax.plot(x, y, color="#37474f", lw=1.0)
        tf = rd[rd.dataset == ds].true_final.iloc[0]
        ax.scatter([500], [tf], marker="*", s=70, color="#37474f", zorder=4)
        for obs in (30, 60, 90):
            r = rd[(rd.dataset == ds) & (rd.obs_depth == obs)].iloc[0]
            ax.axvline(obs, color="#90a4ae", lw=0.6, ls=":")
            ax.scatter([500], [r.predicted_val], marker="D", s=26,
                       color=mcol[r.selected_method], zorder=5)
            ax.plot([obs, 500], [r.current_val, r.predicted_val], lw=0.7,
                    ls="--", color=mcol[r.selected_method], alpha=0.75)
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("boosting round $n$")
    axes[0].set_ylabel("validation log-loss")
    axes[1].legend(handles=[
        Line2D([0], [0], color="#37474f", lw=1, label="validation loss"),
        Line2D([0], [0], marker="*", ls="", color="#37474f", label="true $s_{500}$"),
        Line2D([0], [0], marker="D", ls="", color="#e65100",
               label="prediction (richardson_1)"),
        Line2D([0], [0], marker="D", ls="", color="#2e7d32",
               label="prediction (rational_fit)")],
        fontsize=6.6, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_real_curves.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig1(args.results, args.out)
    fig2(args.results, args.out)
    print(f"Figures written to {args.out}/")

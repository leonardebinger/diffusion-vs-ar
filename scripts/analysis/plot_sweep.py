"""
Analysis + plots for the MDM Sudoku data-efficiency sweeps.

Input: two sweep directories (baseline, rule-loss) containing <tag>/trainer_state.json
and <tag>/sudoku_test/predict_results.json for each N in {N1k, N3k, N10k, N30k, Nfull}.

Output (written next to this script or at --out):
  - fig1_efficiency_curve.png      final test_acc vs N (two curves)
  - fig2_eval_acc_vs_step.png      eval_acc trajectories, faceted by N
  - fig3_rule_loss_vs_step.png     rule_loss + ce_loss over training (rule-loss runs only)
  - summary.csv                    machine-readable table of all metrics
  - equivalence_table.txt          data-equivalence ratios at {50%, 90%, 99%} accuracy

Usage:
    python scripts/analysis/plot_sweep.py \\
        --baseline ~/logs/sudoku-mdm-sweep-baseline-...  \\
        --ruleloss ~/logs/sudoku-mdm-sweep-ruleloss-...  \\
        --out ./figures/

The N size for each tag is parsed from the tag suffix (N1k → 1000, Nfull → 100000).
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TAG_TO_N = {
    "N1k":    1_000,
    "N3k":    3_000,
    "N10k":   10_000,
    "N30k":   30_000,
    "Nfull":  100_000,
}

# Backwards-compat with the older Ndiv* naming
TAG_TO_N.update({
    "Ndiv100":  1_000,
    "Ndiv10":  10_000,
})


def load_run(run_dir: Path):
    """Return dict with keys: n, test_acc, eval_curve=[(step, eval_acc)], train_curve=[(step, ce_loss, rule_loss)]."""
    tag = run_dir.name
    n = TAG_TO_N.get(tag)
    if n is None:
        raise ValueError(f"unknown tag: {tag}")

    pred_path = run_dir / "sudoku_test" / "predict_results.json"
    if pred_path.exists():
        test_acc = json.loads(pred_path.read_text()).get("predict_acc", None)
    else:
        test_acc = None

    state_path = run_dir / "trainer_state.json"
    eval_curve, train_curve = [], []
    if state_path.exists():
        h = json.loads(state_path.read_text()).get("log_history", [])
        for e in h:
            if "eval_acc" in e:
                eval_curve.append((e["step"], e["eval_acc"], e.get("eval_loss")))
            if "loss" in e and "eval_loss" not in e:
                train_curve.append((
                    e["step"],
                    e.get("ce_loss"),
                    e.get("rule_loss"),
                    e["loss"],
                ))
    return dict(tag=tag, n=n, test_acc=test_acc, eval_curve=eval_curve, train_curve=train_curve)


def load_sweep(sweep_dir: Path):
    """Return list of run dicts, sorted by N."""
    runs = []
    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name not in TAG_TO_N:
            continue
        try:
            runs.append(load_run(child))
        except Exception as e:
            print(f"skip {child.name}: {e}", file=sys.stderr)
    runs.sort(key=lambda r: r["n"])
    return runs


def fig1_efficiency_curve(baseline, ruleloss, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for runs, label, marker in [(baseline, "baseline", "o"), (ruleloss, "rule-loss", "s")]:
        xs = [r["n"] for r in runs if r["test_acc"] is not None]
        ys = [r["test_acc"] for r in runs if r["test_acc"] is not None]
        ax.plot(xs, ys, label=label, marker=marker)
    ax.set_xscale("log")
    ax.set_xlabel("Training examples (N)")
    ax.set_ylabel("Test accuracy (whole-puzzle)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("Sudoku MDM data efficiency — baseline vs. rule-loss")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def fig2_eval_curves(baseline, ruleloss, out: Path):
    # Pair runs by N
    by_n = {}
    for r in baseline:
        by_n.setdefault(r["n"], {})["baseline"] = r
    for r in ruleloss:
        by_n.setdefault(r["n"], {})["ruleloss"] = r

    ns = sorted(by_n.keys())
    n_facets = len(ns)
    cols = min(3, n_facets)
    rows = (n_facets + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for ax, n in zip(axes.flatten(), ns):
        for key, style in [("baseline", dict(color="C0", linestyle="-")),
                           ("ruleloss", dict(color="C1", linestyle="-"))]:
            r = by_n[n].get(key)
            if r is None or not r["eval_curve"]:
                continue
            xs = [s for s, a, _ in r["eval_curve"]]
            ys = [a for _, a, _ in r["eval_curve"]]
            ax.plot(xs, ys, label=key, **style)
        ax.set_title(f"N = {n}")
        ax.set_xlabel("step")
        ax.set_ylabel("eval_acc")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes.flatten()[n_facets:]:
        ax.axis("off")

    fig.suptitle("Validation accuracy over training")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def fig3_rule_loss(ruleloss, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_ce, ax_rule = axes
    for r in ruleloss:
        tc = r["train_curve"]
        if not tc:
            continue
        steps = [s for s, ce, rl, tot in tc]
        ce_vals = [ce for _, ce, _, _ in tc if ce is not None]
        rl_vals = [rl for _, _, rl, _ in tc if rl is not None]
        label = f'N={r["n"]}'
        if ce_vals:
            ax_ce.plot(steps[:len(ce_vals)], ce_vals, label=label, alpha=0.7)
        if rl_vals:
            ax_rule.plot(steps[:len(rl_vals)], rl_vals, label=label, alpha=0.7)

    ax_ce.set_title("ce_loss over training (rule-loss runs)")
    ax_ce.set_xlabel("step"); ax_ce.set_ylabel("ce_loss")
    ax_ce.set_yscale("log"); ax_ce.grid(True, alpha=0.3); ax_ce.legend(fontsize=8)

    ax_rule.set_title("rule_loss over training")
    ax_rule.set_xlabel("step"); ax_rule.set_ylabel("rule_loss")
    ax_rule.grid(True, alpha=0.3); ax_rule.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def write_summary_csv(baseline, ruleloss, out: Path):
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "tag", "N", "test_acc", "best_eval_acc"])
        for cond, runs in [("baseline", baseline), ("ruleloss", ruleloss)]:
            for r in runs:
                best = max((a for _, a, _ in r["eval_curve"]), default=None)
                w.writerow([cond, r["tag"], r["n"], r["test_acc"], best])
    print(f"wrote {out}")


def equivalence_table(baseline, ruleloss, out: Path,
                      thresholds=(0.50, 0.90, 0.99)):
    """For each threshold, find smallest N in each condition reaching it."""
    def first_n_at(runs, thr):
        for r in sorted(runs, key=lambda r: r["n"]):
            if r["test_acc"] is not None and r["test_acc"] >= thr:
                return r["n"]
        return None

    lines = ["threshold | baseline_N | ruleloss_N | data_reduction_x"]
    lines.append("-" * 60)
    for thr in thresholds:
        b = first_n_at(baseline, thr)
        r = first_n_at(ruleloss, thr)
        if b and r and r > 0:
            ratio = f"{b / r:.2f}x"
        else:
            ratio = "n/a"
        lines.append(f"  {thr:>5.0%}  |  {str(b or '-'):>9}  |  {str(r or '-'):>9}  |  {ratio}")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--ruleloss", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("./figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    baseline = load_sweep(args.baseline)
    ruleloss = load_sweep(args.ruleloss)
    print(f"baseline: {len(baseline)} runs | ruleloss: {len(ruleloss)} runs")

    write_summary_csv(baseline, ruleloss, args.out / "summary.csv")
    fig1_efficiency_curve(baseline, ruleloss, args.out / "fig1_efficiency_curve.png")
    fig2_eval_curves(baseline, ruleloss, args.out / "fig2_eval_acc_vs_step.png")
    fig3_rule_loss(ruleloss, args.out / "fig3_rule_loss_vs_step.png")
    equivalence_table(baseline, ruleloss, args.out / "equivalence_table.txt")


if __name__ == "__main__":
    main()

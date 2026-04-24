"""
Plot baseline vs. permanent-rule-loss comparison for N ∈ {1000, 3000, 10000}.

Produces 4 figures in the output directory:
  fig1_eval_acc_N1000.png      side-by-side: baseline | permanent at N=1000
  fig2_eval_acc_N3000.png      side-by-side: baseline | permanent at N=3000
  fig3_eval_acc_N10000.png     side-by-side: baseline | permanent at N=10000
  fig4_test_acc_bars.png       bar chart: test_acc vs N, baseline vs permanent

Usage:
  python scripts/analysis/plot_permanent_vs_baseline.py \
      --baseline-N1000  /path/to/N1k/run \
      --baseline-N3000  /path/to/N3k/run \
      --baseline-N10000 /path/to/N10k/run \
      --permanent-root  /path/to/sudoku-mdm-sweep-ruleloss-...smallN.../ \
      --out             ./figures

The --permanent-root is the sweep directory (it must contain subdirs
N1000/, N3000/, N10000/).

Each run dir must contain:
  trainer_state.json                    (for eval_acc trajectories)
  sudoku_test/predict_results.json      (for final test_acc)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run(run_dir: Path):
    run_dir = Path(run_dir)
    state = json.loads((run_dir / "trainer_state.json").read_text())
    evals = [(e["step"], e["eval_acc"], e["eval_loss"])
             for e in state["log_history"] if "eval_acc" in e]
    pred_path = run_dir / "sudoku_test" / "predict_results.json"
    test_acc = None
    if pred_path.exists():
        test_acc = json.loads(pred_path.read_text()).get("predict_acc")
    return {"dir": run_dir, "evals": evals, "test_acc": test_acc}


def fig_side_by_side(baseline, permanent, N: int, out_path: Path):
    """Left: baseline eval_acc vs step. Right: permanent eval_acc vs step."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, run, title, color in [
        (ax_l, baseline, f"baseline, N={N}", "C0"),
        (ax_r, permanent, f"permanent rule loss, N={N}", "C1"),
    ]:
        if run is None or not run["evals"]:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            steps = [s for s, _, _ in run["evals"]]
            accs = [a for _, a, _ in run["evals"]]
            ax.plot(steps, accs, color=color, linewidth=1.2)
            if run.get("test_acc") is not None:
                ax.axhline(run["test_acc"], color=color, linestyle="--", alpha=0.5,
                           label=f'test_acc = {run["test_acc"]:.3f}')
                ax.legend(loc="lower right", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    ax_l.set_ylabel("eval accuracy (whole-puzzle)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_test_acc_bars(baselines: dict, permanents: dict, Ns: list, out_path: Path):
    """Bar chart: test_acc vs N, side-by-side baseline and permanent."""
    x = np.arange(len(Ns))
    w = 0.38
    base_accs = [baselines[n]["test_acc"] if baselines.get(n) else 0 for n in Ns]
    perm_accs = [permanents[n]["test_acc"] if permanents.get(n) else 0 for n in Ns]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_b = ax.bar(x - w / 2, base_accs, width=w, color="C0", label="baseline")
    bars_p = ax.bar(x + w / 2, perm_accs, width=w, color="C1", label="permanent rule loss")

    for bars, accs in [(bars_b, base_accs), (bars_p, perm_accs)]:
        for bar, acc in zip(bars, accs):
            if acc is None:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N = {n}" for n in Ns])
    ax.set_ylabel("test accuracy (whole-puzzle, held-out 1000 puzzles)")
    ax.set_ylim(0, 1.08)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Sudoku MDM: baseline vs permanent rule loss — final test accuracy")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-N1000",  required=True, type=Path)
    ap.add_argument("--baseline-N3000",  required=True, type=Path)
    ap.add_argument("--baseline-N10000", required=True, type=Path)
    ap.add_argument("--permanent-root",  required=True, type=Path,
                    help="sweep directory containing N1000/, N3000/, N10000/ subdirs")
    ap.add_argument("--out", type=Path, default=Path("./figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    baselines = {
        1000:  load_run(args.baseline_N1000),
        3000:  load_run(args.baseline_N3000),
        10000: load_run(args.baseline_N10000),
    }
    permanents = {
        1000:  load_run(args.permanent_root / "N1000"),
        3000:  load_run(args.permanent_root / "N3000"),
        10000: load_run(args.permanent_root / "N10000"),
    }

    # Per-N side-by-side figures.
    for N, fname in [(1000, "fig1_eval_acc_N1000.png"),
                     (3000, "fig2_eval_acc_N3000.png"),
                     (10000, "fig3_eval_acc_N10000.png")]:
        fig_side_by_side(baselines[N], permanents[N], N, args.out / fname)

    # Overall test_acc bar chart.
    fig_test_acc_bars(baselines, permanents, [1000, 3000, 10000],
                      args.out / "fig4_test_acc_bars.png")

    # One-line summary table.
    print()
    print(f"{'N':>6}  {'baseline':>10}  {'permanent':>10}  {'Δ':>8}")
    print("-" * 40)
    for N in (1000, 3000, 10000):
        b = baselines[N]["test_acc"]; p = permanents[N]["test_acc"]
        d = (p - b) if (b is not None and p is not None) else None
        print(f"{N:>6}  {b if b is not None else '—':>10}  "
              f"{p if p is not None else '—':>10}  "
              f"{d if d is not None else '—':>+8.3f}" if d is not None
              else f"{N:>6}  {b}  {p}  —")


if __name__ == "__main__":
    main()

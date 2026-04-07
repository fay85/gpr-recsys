"""Plot MTP and VAFT train loss curves, one figure each, comparing all runs."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

PLOTS = [
    ("MTP Train Loss", "mtp_train_epoch/avg_loss", "mtp_train_loss.png"),
    ("VAFT Train Loss", "vaft_train_epoch/avg_loss", "vaft_train_loss.png"),
]

STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]


def main():
    run_dirs = sorted(
        d for d in os.listdir(RUNS_DIR)
        if os.path.isdir(os.path.join(RUNS_DIR, d))
    )
    n_runs = len(run_dirs)
    print(f"Found {n_runs} runs:")
    for r in run_dirs:
        print(f"  {r}")

    accumulators = {}
    for rd in run_dirs:
        ea = EventAccumulator(os.path.join(RUNS_DIR, rd))
        ea.Reload()
        accumulators[rd] = ea

    short_labels = [
        rd.split("_")[-2] + "_" + rd.split("_")[-1] for rd in run_dirs
    ]

    colors = [cm.tab10(i / max(n_runs - 1, 1)) for i in range(n_runs)]

    for title, tag, filename in PLOTS:
        fig, ax = plt.subplots(figsize=(10, 6))
        for run_idx, (rd, ea) in enumerate(accumulators.items()):
            if tag not in ea.Tags().get("scalars", []):
                print(f"  [skip] {rd} has no tag '{tag}'")
                continue
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            vals = [e.value for e in events]
            ax.plot(
                steps, vals,
                color=colors[run_idx],
                linestyle=STYLES[run_idx % len(STYLES)],
                linewidth=1.8,
                label=short_labels[run_idx],
                alpha=0.85,
            )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(RUNS_DIR, filename)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

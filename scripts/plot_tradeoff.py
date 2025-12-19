"""
Generate the accuracy vs. compression trade-off plot.

By default, this script plots the reported numbers from the paper/README.
Optionally, pass a CSV produced by run_sweep.py to plot actual run outputs.

Examples:
  python scripts/plot_tradeoff.py
  python scripts/plot_tradeoff.py --csv outputs/results/results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


DEFAULT_POINTS: List[Tuple[str, float, float]] = [
    ("Baseline", 1.00, 88.14),
    ("K-Means k=16", 8.00, 28.06),
    ("K-Means k=32", 6.40, 25.62),
    ("K-Means k=128", 4.57, 34.18),
    ("GMM 16", 8.00, 20.00),
    ("GMM 32", 6.40, 15.42),
    ("DBSCAN eps=0.1", 32.00, 9.80),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="", help="Optional CSV from outputs/results/results.csv")
    p.add_argument("--out-dir", type=str, default=".", help="Where to write fig_tradeoff.[png|pdf]")
    return p.parse_args()


def load_points_from_csv(path: str) -> List[Tuple[str, float, float]]:
    points = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["Method"]
            hp = row["Hyperparameter"]
            val = row["Value"]
            label = f"{method} {hp}={val}" if method != "Baseline" else "Baseline"
            x = float(row["Compression Ratio"])
            y = float(row["Test Accuracy"])
            points.append((label, x, y))
    return points


def main() -> None:
    args = parse_args()
    points = DEFAULT_POINTS
    if args.csv:
        points = load_points_from_csv(args.csv)

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required. Install it (pip install matplotlib) and re-run.") from e

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    labels = [p[0] for p in points]

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, s=60)
    for x, y, label in zip(xs, ys, labels):
        plt.text(x * 1.03, y, label, fontsize=8, va="center")
    plt.xscale("log", base=2)
    plt.xlabel("Compression ratio (x) (higher means smaller model)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Accuracy vs. Compression Ratio (ResNet-18 on CIFAR-10)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "fig_tradeoff.pdf"
    out_png = out_dir / "fig_tradeoff.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=200)
    print(f"Saved: {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()

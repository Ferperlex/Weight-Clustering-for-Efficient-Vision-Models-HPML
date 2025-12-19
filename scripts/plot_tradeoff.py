"""
Generate the accuracy vs. compression trade-off plot from the reported results.

This script is intentionally standalone and does NOT retrain the model or
modify any artifacts. It only plots the already-reported numbers.
"""

def main() -> None:
    # Reported results (from README / paper table)
    points = [
        ("Baseline",         1.00, 88.14),
        ("K-Means k=16",     8.00, 28.06),
        ("K-Means k=32",     6.40, 25.62),
        ("K-Means k=128",    4.57, 34.18),
        ("GMM 16",           8.00, 20.00),
        ("GMM 32",           6.40, 15.42),
        ("DBSCAN eps=0.1",  32.00,  9.80),
    ]

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required to generate the plot. "
            "Install it (e.g., pip install matplotlib) and re-run."
        ) from e

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    labels = [p[0] for p in points]

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, s=60)
    for x, y, label in zip(xs, ys, labels):
        plt.text(x * 1.03, y, label, fontsize=8, va="center")
    plt.xscale("log", base=2)
    plt.xlabel("Compression ratio (×) (higher means smaller model)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Accuracy vs. Compression Ratio (ResNet-18 on CIFAR-10)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_pdf = "fig_tradeoff.pdf"
    out_png = "fig_tradeoff.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    print(f"Saved: {out_pdf}, {out_png}")

if __name__ == "__main__":
    main()

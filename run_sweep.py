from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.compression.clustering import apply_kmeans, apply_gmm, apply_dbscan
from src.compression.pipeline import cluster_convs_inplace, estimate_final_size_mb
from src.data.cifar10 import get_cifar10_loaders
from src.models.resnet_cifar import ResNet18
from src.training.engine import train, evaluate, get_model_size_mb
from src.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce full notebook run: baseline + clustering sweeps + summary plot.")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-batch-size", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--epochs-baseline", type=int, default=20)
    p.add_argument("--baseline-lr", type=float, default=0.01)
    p.add_argument("--baseline-momentum", type=float, default=0.9)
    p.add_argument("--baseline-weight-decay", type=float, default=5e-4)

    p.add_argument("--epochs-finetune", type=int, default=20)
    p.add_argument("--ft-lr", type=float, default=0.001)
    p.add_argument("--ft-momentum", type=float, default=0.9)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--baseline-checkpoint", type=str, default="baseline_resnet18.pth")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="resnet-compression")

    p.add_argument("--out-dir", type=str, default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    print(f"Running on: {device}")

    out_dir = Path(args.out_dir)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    loaders = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    # ---------------- Baseline ----------------
    wandb_module = None
    if args.wandb:
        import wandb
        wandb_module = wandb
        wandb_module.init(project=args.wandb_project, name="baseline", reinit=True)

    print("--- Phase 1: Baseline Training ---")
    baseline_model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        baseline_model.parameters(),
        lr=args.baseline_lr,
        momentum=args.baseline_momentum,
        weight_decay=args.baseline_weight_decay,
    )

    train(baseline_model, loaders.trainloader, optimizer, criterion, args.epochs_baseline, device, wandb_module=wandb_module)
    baseline_acc = evaluate(baseline_model, loaders.testloader, device)
    baseline_size_mb = get_model_size_mb(baseline_model)
    print(f"Baseline Accuracy: {baseline_acc:.2f}% | Size: {baseline_size_mb:.2f} MB")

    if wandb_module is not None:
        wandb_module.log({"Test Accuracy": baseline_acc, "Model Size (MB)": baseline_size_mb})
        wandb_module.finish()

    torch.save(baseline_model.state_dict(), args.baseline_checkpoint)

    # ---------------- Experiments ----------------
    def run_one(method_name: str, param_name: str, param_val, cluster_func):
        run_name = f"{method_name}_{param_val}"
        w = None
        if args.wandb:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={"method": method_name, param_name: param_val, "epochs_finetune": args.epochs_finetune},
                reinit=True,
            )
            w = wandb

        model = ResNet18().to(device)
        model.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device))

        stats, _ = cluster_convs_inplace(model, cluster_func, param_val, device, rng=rng)

        ft_optimizer = optim.SGD(model.parameters(), lr=args.ft_lr, momentum=args.ft_momentum)
        print("Fine-tuning centroids...")
        train(model, loaders.trainloader, ft_optimizer, criterion, args.epochs_finetune, device, wandb_module=w)

        acc = evaluate(model, loaders.testloader, device)
        comp_ratio, final_size_mb = estimate_final_size_mb(
            baseline_size_mb=baseline_size_mb,
            conv_original_bits=stats.total_original_bits,
            conv_compressed_bits=stats.total_compressed_bits,
        )

        print(f"Result: Acc: {acc:.2f}% | Ratio: {comp_ratio:.2f}x | Size: {final_size_mb:.2f} MB")

        if w is not None:
            w.log({"Test Accuracy": acc, "Compression Ratio": comp_ratio, "Model Size (MB)": final_size_mb})
            w.finish()

        return {
            "Method": method_name,
            "Hyperparameter": param_name,
            "Value": param_val,
            "Test Accuracy": round(acc, 2),
            "Compression Ratio": round(comp_ratio, 2),
            "Model Size (MB)": round(final_size_mb, 2),
        }

    results = []
    for k in [16, 32, 128]:
        results.append(run_one("K-Means", "k", k, apply_kmeans))
    for n in [16, 32]:
        results.append(run_one("GMM", "Components", n, apply_gmm))
    for eps in [0.1, 0.2]:
        results.append(run_one("DBSCAN", "Eps", eps, apply_dbscan))

    csv_path = out_dir / "results" / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to: {csv_path}")

    # Summary plot (matches notebook)
    import matplotlib.pyplot as plt

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name="summary_analysis", reinit=True)

    methods = list(set([r["Method"] for r in results]))
    plt.figure(figsize=(10, 6))
    for method in methods:
        x = [r["Compression Ratio"] for r in results if r["Method"] == method]
        y = [r["Test Accuracy"] for r in results if r["Method"] == method]
        plt.scatter(x, y, label=method, s=100, alpha=0.7)

    plt.xscale("log", base=2)
    plt.xlabel("Compression Ratio (Higher Means Smaller Model)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs. Compression Ratio")
    plt.legend()
    plt.grid(True)

    fig_png = out_dir / "figures" / "fig_tradeoff.png"
    fig_pdf = out_dir / "figures" / "fig_tradeoff.pdf"
    plt.savefig(fig_png, bbox_inches="tight", dpi=200)
    plt.savefig(fig_pdf, bbox_inches="tight")
    print(f"Saved plots: {fig_png}, {fig_pdf}")

    if args.wandb:
        wandb.log({"Accuracy vs Compression": wandb.Image(plt)})
        wandb.finish()

    print("--- Final Summary ---")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()

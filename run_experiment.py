from __future__ import annotations

import argparse
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
    p = argparse.ArgumentParser(description="Cluster all convolutional weights, fine-tune centroids, and evaluate.")
    p.add_argument("--baseline-checkpoint", type=str, default="baseline_resnet18.pth")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-batch-size", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--epochs-finetune", type=int, default=20)
    p.add_argument("--ft-lr", type=float, default=0.001)
    p.add_argument("--ft-momentum", type=float, default=0.9)

    p.add_argument("--method", type=str, required=True, choices=["kmeans", "gmm", "dbscan"])
    p.add_argument("--param", type=float, required=True)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="resnet-compression")
    p.add_argument("--wandb-run-name", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    print(f"Running on: {device}")

    loaders = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    # Load baseline to estimate non-conv size consistently
    baseline_model = ResNet18().to(device)
    baseline_model.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device))
    baseline_size_mb = get_model_size_mb(baseline_model)

    # Set clustering function
    if args.method == "kmeans":
        cluster_func = apply_kmeans
        method_name = "K-Means"
        param_name = "k"
        param_val = int(args.param)
    elif args.method == "gmm":
        cluster_func = apply_gmm
        method_name = "GMM"
        param_name = "Components"
        param_val = int(args.param)
    else:
        cluster_func = apply_dbscan
        method_name = "DBSCAN"
        param_name = "Eps"
        param_val = float(args.param)

    run_name = args.wandb_run_name or f"{method_name}_{param_val}"

    wandb_module = None
    if args.wandb:
        import wandb
        wandb_module = wandb
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={"method": method_name, param_name: param_val, "epochs_finetune": args.epochs_finetune},
            reinit=True,
        )

    # Clone model and load baseline weights
    model = ResNet18().to(device)
    model.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device))

    # Cluster and replace convolution layers
    stats, _ = cluster_convs_inplace(model, cluster_func, param_val, device, rng=rng)

    # Fine-tune
    criterion = nn.CrossEntropyLoss()
    ft_optimizer = optim.SGD(model.parameters(), lr=args.ft_lr, momentum=args.ft_momentum)
    print("Fine-tuning centroids...")
    train(model, loaders.trainloader, ft_optimizer, criterion, args.epochs_finetune, device, wandb_module=wandb_module)

    # Evaluate
    acc = evaluate(model, loaders.testloader, device)

    # Size estimate and ratio (matches notebook)
    comp_ratio, final_size_mb = estimate_final_size_mb(
        baseline_size_mb=baseline_size_mb,
        conv_original_bits=stats.total_original_bits,
        conv_compressed_bits=stats.total_compressed_bits,
    )

    print(f"Result: Acc: {acc:.2f}% | Ratio: {comp_ratio:.2f}x | Size: {final_size_mb:.2f} MB")

    if wandb_module is not None:
        wandb_module.log({"Test Accuracy": acc, "Compression Ratio": comp_ratio, "Model Size (MB)": final_size_mb})
        wandb_module.finish()

    # Save clustered checkpoint if desired (optional, does not affect metrics)
    # Keeping default behavior as notebook (no clustered checkpoint saved).


if __name__ == "__main__":
    main()

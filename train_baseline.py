from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.cifar10 import get_cifar10_loaders
from src.models.resnet_cifar import ResNet18
from src.training.engine import train, evaluate, get_model_size_mb
from src.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline ResNet-18 on CIFAR-10 (matches notebook defaults).")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-batch-size", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--save-path", type=str, default="baseline_resnet18.pth")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="resnet-compression")
    p.add_argument("--wandb-run-name", type=str, default="baseline")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    print(f"Running on: {device}")

    loaders = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    wandb_module = None
    if args.wandb:
        import wandb
        wandb_module = wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, reinit=True)

    train(model, loaders.trainloader, optimizer, criterion, args.epochs, device, wandb_module=wandb_module)

    test_acc = evaluate(model, loaders.testloader, device)
    size_mb = get_model_size_mb(model)
    print(f"Baseline Accuracy: {test_acc:.2f}% | Size: {size_mb:.2f} MB")

    if wandb_module is not None:
        wandb_module.log({"Test Accuracy": test_acc, "Model Size (MB)": size_mb})
        wandb_module.finish()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_path)


if __name__ == "__main__":
    main()

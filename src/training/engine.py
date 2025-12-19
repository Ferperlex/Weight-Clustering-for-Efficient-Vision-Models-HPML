from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def train(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
    wandb_module: Optional[object] = None,
) -> None:
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(loader)
        epoch_acc = 100.0 * correct / total

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.3f} | Acc: {epoch_acc:.2f}%")

        if wandb_module is not None:
            wandb_module.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "train_accuracy": epoch_acc,
                }
            )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def get_model_size_mb(model: torch.nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / (1024**2)

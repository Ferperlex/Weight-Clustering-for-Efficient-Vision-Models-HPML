from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .clustered_layers import ClusteredConv2d


@dataclass
class CompressionStats:
    total_original_bits: float
    total_compressed_bits: float


def cluster_convs_inplace(
    model: nn.Module,
    cluster_func: Callable,
    param_val,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[CompressionStats, List[Tuple[str, nn.Module]]]:
    total_original_bits = 0.0
    total_compressed_bits = 0.0
    modules_to_replace: List[Tuple[str, nn.Module]] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weights = module.weight.data.cpu().numpy().flatten()

            centroids, labels = cluster_func(weights, param_val) if cluster_func.__name__ != "apply_dbscan" else cluster_func(weights, param_val, rng=rng)

            clustered_layer = ClusteredConv2d(module, centroids, labels)
            modules_to_replace.append((name, clustered_layer))

            n_weights = len(weights)
            n_centroids = len(centroids)

            total_original_bits += n_weights * 32

            n_centroids = max(1, n_centroids)
            bits_per_index = np.ceil(np.log2(n_centroids)) if n_centroids > 1 else 1
            total_compressed_bits += (n_centroids * 32) + (n_weights * bits_per_index)

    # Replace layers after iteration
    for name, new_layer in modules_to_replace:
        sub_module = model
        tokens = name.split(".")
        for token in tokens[:-1]:
            sub_module = getattr(sub_module, token)
        setattr(sub_module, tokens[-1], new_layer.to(device))

    return CompressionStats(total_original_bits=total_original_bits, total_compressed_bits=total_compressed_bits), modules_to_replace


def estimate_final_size_mb(
    baseline_size_mb: float,
    conv_original_bits: float,
    conv_compressed_bits: float,
) -> Tuple[float, float]:
    comp_ratio = conv_original_bits / conv_compressed_bits if conv_compressed_bits > 0 else 0.0

    final_size_mb = (conv_compressed_bits / 8) / (1024**2)

    non_conv_size = baseline_size_mb - (conv_original_bits / 8 / 1024**2)
    final_size_mb += non_conv_size

    return comp_ratio, final_size_mb

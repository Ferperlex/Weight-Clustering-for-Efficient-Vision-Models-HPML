from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusteredConv2d(nn.Module):
    def __init__(self, original_conv: nn.Conv2d, centroids, labels) -> None:
        super().__init__()

        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups

        self.bias = nn.Parameter(original_conv.bias) if original_conv.bias is not None else None

        # Centroids are trainable parameters
        self.centroids = nn.Parameter(torch.tensor(centroids, dtype=torch.float32))

        # Indices are frozen
        self.register_buffer("weight_indices", torch.tensor(labels, dtype=torch.long))

        self.weight_shape = original_conv.weight.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstructed_weight = self.centroids[self.weight_indices].view(self.weight_shape)
        return F.conv2d(
            x,
            reconstructed_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

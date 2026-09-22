from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpatialClassSemanticEncoder(nn.Module):
    """Convert a class-index segmentation map into a bottleneck feature map."""

    def __init__(
        self,
        *,
        num_classes: int = 5,
        embedding_dim: int = 32,
        output_channels: int = 512,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.output_channels = int(output_channels)
        self.embedding = nn.Embedding(self.num_classes, self.embedding_dim)
        self.projection = nn.Conv2d(self.embedding_dim, self.output_channels, kernel_size=1)
        self.reset_projection()

    def reset_projection(self) -> None:
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        segmentation: torch.Tensor,
        *,
        spatial_size: tuple[int, int],
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if segmentation.ndim != 4 or segmentation.shape[1] != 1:
            raise ValueError(
                "segmentation must have shape [B,1,H,W], got "
                f"{tuple(segmentation.shape)}"
            )
        rounded = segmentation.round()
        if not torch.equal(segmentation, rounded):
            raise ValueError("segmentation must contain integer class indices")
        indices = rounded[:, 0].to(dtype=torch.long)
        if indices.numel() and (indices.min().item() < 0 or indices.max().item() >= self.num_classes):
            raise ValueError(f"segmentation values must be in [0,{self.num_classes - 1}]")

        embedded = self.embedding(indices).permute(0, 3, 1, 2).contiguous()
        embedded = F.interpolate(embedded, size=spatial_size, mode="nearest")
        projected = self.projection(embedded)
        return projected.to(dtype=output_dtype)

    def projection_is_zero(self) -> bool:
        return bool(
            torch.count_nonzero(self.projection.weight).item() == 0
            and torch.count_nonzero(self.projection.bias).item() == 0
        )

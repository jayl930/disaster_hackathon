from typing import List, Literal
from utils.utils_manipulate_layers import SwapDim
from utils.utils_iterative_upsample import (
    IterativeUpsample,
    IterativeUpsampleStepParams,
)
import torch.nn as nn
import torch
from einops import rearrange


class Head(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        decoder_depth: int,
        img_size: List[int],
        patch_size: int,
        upsampling_steps: List[IterativeUpsampleStepParams],
        out_dim: int,
        feature_extractor_type: Literal["simple", "res-net"],
        double=False,
    ) -> None:
        super(Head, self).__init__()

        img_patched_dim = [
            img_size[0] // patch_size,
            img_size[1] // patch_size,
        ]
        self.is_double = double

        self.decoder_layers: nn.Sequential = nn.Sequential()
        if double:
            self.double_proj = nn.Linear(embed_dim * 2, embed_dim)
        for _ in range(decoder_depth):
            self.decoder_layers.append(nn.Linear(embed_dim, embed_dim))
            self.decoder_layers.append(nn.GELU())

        # (batch #, patch #, embedding dim) -> (batch #, embedding dim, patch #)
        self.swap_layer_one = SwapDim(1, 2)

        # Reshape into 2d image: (batch #, embedding dim, patch #) -> (batch #, embedding dim, patch_i, patch_j)
        self.img_patched_dim = img_patched_dim
        self.reshape_to_3D = nn.Unflatten(2, (img_patched_dim[0], img_patched_dim[1]))

        # Iterative upscaling + convolution
        self.iterative_upsample = IterativeUpsample(
            params=upsampling_steps,
            original_channel_dim=embed_dim,
            out_dim=out_dim,
            feature_extractor_type=feature_extractor_type,
            double=double,
        )

        # (batch #, 1, img_dim[0], img_dim[1]) -> (batch #, patch_size * patch_size * out_dim, patch #)
        self.group_into_patches = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        # (batch #, patch_size * patch_size * out_dim, patch #) -> (batch #, patch #, patch_size * patch_size * out_dim)
        self.swap_layer_two = SwapDim(1, 2)

    def forward(self, x: torch.Tensor, original_image: torch.Tensor):
        if self.is_double:
            x = self.double_proj(x)
        x = self.decoder_layers.forward(x)  # N L C
        x = rearrange(
            x,
            "n (h w) c -> n c h w",
            h=self.img_patched_dim[0],
            w=self.img_patched_dim[1],
        )
        # x = self.swap_layer_one(x)
        # x = self.reshape_to_3D.forward(x) * 0.0
        x = self.iterative_upsample.forward(x, original_image)
        x = self.group_into_patches.forward(x)
        return self.swap_layer_two.forward(x)

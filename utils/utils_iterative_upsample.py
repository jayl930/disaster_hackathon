from typing import List, Literal, Optional, TypedDict
import torch.nn as nn
import torch
from utils.utils_feature_extractor import (
    FeatureExtractor,
    IterativeUpsampleStepParams,
    ResNet50FeatureExtractor,
)
from utils.utils_resize_conv import ResizeConv2D


def _create_feature_extractor(
    params: List[IterativeUpsampleStepParams], original_feature_channel_dim: int
):
    # Create scale_factor list to be provided to FeatureExtractor
    downscale_factors = [step_param["step_scale_factor"] for step_param in params]

    # Reverse scale factors so that each dimension gets the correct channel depth (instead of small to large, feature extractor goes from large to small)
    downscale_factors.reverse()

    # Create channels list
    feature_dims = []
    for step_param in params:
        feature_dims.append(step_param["feature_dim"])
    feature_dims.append(original_feature_channel_dim)
    # Reverse for same reason we reverse downscale_factors
    feature_dims.reverse()

    return FeatureExtractor(channels=feature_dims, down_sample_rates=downscale_factors)


class IterativeUpsample(nn.Module):
    def __init__(
        self,
        params: List[IterativeUpsampleStepParams],
        original_channel_dim,
        feature_extractor_type: Literal["simple", "res-net"],
        original_feature_channel_dim=3,
        out_dim=1,
        double=False,
    ):
        super(IterativeUpsample, self).__init__()

        if feature_extractor_type == "simple":
            self.feature_extractor = _create_feature_extractor(
                params=params, original_feature_channel_dim=original_feature_channel_dim
            )
        else:
            self.feature_extractor = ResNet50FeatureExtractor(upsampling_steps=params)

        self.resize_conv_layers = nn.ModuleList()
        self.is_double = double

        old_embed_dim = original_channel_dim
        for param in params:
            scale_factor = param["step_scale_factor"]
            new_channel_dim = param["new_channel_dim"]
            feature_dim = param["feature_dim"]
            block_count = param.get("block_count")
            if self.is_double:
                input_dim = old_embed_dim + feature_dim * 2
            else:
                input_dim = old_embed_dim + feature_dim
            self.resize_conv_layers.append(
                ResizeConv2D(
                    scale_factor=scale_factor,
                    old_channel_size=input_dim,
                    new_channel_size=new_channel_dim,
                    layer_norm_after_upsample=(old_embed_dim == original_channel_dim),
                    block_count=block_count if block_count is not None else 1,
                )
            )
            old_embed_dim = new_channel_dim
        self.cls_head = nn.Conv2d(new_channel_dim, out_dim, 1)

    def forward(self, x: torch.Tensor, original_image: torch.Tensor):
        if self.is_double:
            x0, x1 = original_image.chunk(2, dim=1)
            original_image = torch.cat([x0, x1])
            feature_map = self.feature_extractor.forward(original_image)
            feature_map.reverse()
            for i in range(len(feature_map)):
                x0, x1 = feature_map[i].chunk(2, dim=0)
                feature_map[i] = torch.cat([x0, x1], dim=1)
        else:
            feature_map = self.feature_extractor.forward(original_image)
            # Feature map returns by default with the largest feature first. We want to start with smallest as we upsample
            feature_map.reverse()

        for resize_conv_layer, feature in zip(self.resize_conv_layers, feature_map):
            x = torch.cat((x, feature), dim=1)
            x = resize_conv_layer.forward(x)
        x = self.cls_head(x)
        return x

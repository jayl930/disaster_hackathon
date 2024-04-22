import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_, DropPath


class ResizeConv2D(nn.Module):
    def __init__(
        self,
        scale_factor: int,
        old_channel_size: int,
        new_channel_size: int,
        layer_norm_after_upsample: bool,
        block_count,
    ):
        super(ResizeConv2D, self).__init__()

        self.layers = nn.Sequential()
        # This is based on ConvNext's stem
        if layer_norm_after_upsample:
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=old_channel_size,
                    out_channels=new_channel_size,
                    kernel_size=scale_factor,
                    stride=scale_factor,
                    padding=0,
                )
            )
            self.layers.append(nn.GroupNorm(4, new_channel_size))

        else:
            self.layers.append(nn.GroupNorm(4, old_channel_size))
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=old_channel_size,
                    out_channels=new_channel_size,
                    kernel_size=scale_factor,
                    stride=scale_factor,
                    padding=0,
                )
            )

        for _ in range(block_count):
            self.layers.append(ConvNextBlock(new_channel_size, old_channel_size))

    def forward(self, x: torch.Tensor):
        return self.layers.forward(x)


class LayerNormForChannelInFirstColumn(nn.Module):
    def __init__(self, dim: int) -> None:
        super(LayerNormForChannelInFirstColumn, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        return x.permute(0, 3, 1, 2)


# ConvNeXt Block from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L12
class ConvNextBlock(nn.Module):
    def __init__(
        self,
        input_channel_size: int,
        intermediary_channel_size: int,
        drop_path_rate=0.0,  # Only increase this if training is taking too long
        layer_scale_init_value=1e-6,
    ) -> None:
        super(ConvNextBlock, self).__init__()
        self.depth_conv = nn.Conv2d(
            input_channel_size,
            input_channel_size,
            kernel_size=7,
            padding=3,
            groups=input_channel_size,
        )

        self.layer_norm = nn.LayerNorm(input_channel_size)
        self.linear_to_intermediary_channel_dim = nn.Linear(
            input_channel_size, intermediary_channel_size
        )
        self.gelu = nn.GELU()
        self.linear_to_input_channel_dim = nn.Linear(
            intermediary_channel_size, input_channel_size
        )
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((input_channel_size)),
                requires_grad=True,
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        original_input = x
        x = self.depth_conv.forward(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm.forward(x)
        x = self.linear_to_intermediary_channel_dim.forward(x)
        x = self.gelu.forward(x)
        x = self.linear_to_input_channel_dim.forward(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        return original_input + self.drop_path(x)

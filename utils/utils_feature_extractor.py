from typing import List, Optional, TypedDict
import torch.nn as nn
import torch
from zoo.senet import se_resnext50_32x4d


class IterativeUpsampleStepParams(TypedDict):
    step_scale_factor: int
    new_channel_dim: int
    feature_dim: int
    block_count: Optional[int]


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.is_double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.is_double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, down_sample_rate):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(down_sample_rate),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class FeatureExtractor(nn.Module):
    """
    Example:
    channels = [3, 64, 128, 256, 512]
    down_sample_rates = [1, 2, 2, 2]
    """

    def __init__(self, channels: list[int], down_sample_rates: list[int]):
        super(FeatureExtractor, self).__init__()
        assert len(channels) == len(down_sample_rates) + 1
        self.down_sample_steps = nn.ModuleList()
        self.feature_info = []
        for i in range(len(channels) - 1):
            self.down_sample_steps.append(
                Down(channels[i], channels[i + 1], down_sample_rates[i])
            )
            self.feature_info.append(
                {
                    "in_channels": channels[i],
                    "out_channels": channels[i + 1],
                    "down_sample_rate": down_sample_rates[i],
                }
            )

    """
    x: (batch_size, channels=3, image_size=1024, image_size=1024)
    hidden_states: [
        (bs, 64, 1024, 1024),
        (bs, 128, 512, 512),
        (bs, 256, 256, 256),
        (bs, 512, 1024, 1024),
    ]
    """

    def forward(self, x: torch.Tensor):
        feature_map: List[torch.Tensor] = []
        for down_sample_step in self.down_sample_steps:
            x = down_sample_step.forward(x)
            feature_map.append(x)
        return feature_map


class ResNet50FeatureExtractor(nn.Module):
    """
    no need to pass channels and down_sample_rates
    """

    def __init__(self, upsampling_steps: List[IterativeUpsampleStepParams]):
        super(ResNet50FeatureExtractor, self).__init__()
        encoder = se_resnext50_32x4d(pretrained=None)
        self.layers = nn.ModuleList()

        # Make sure all step scales are 2
        assert all(step["step_scale_factor"] == 2 for step in upsampling_steps)

        # Make sure the number of steps is less than or eqaul to the number of hidden states
        assert len(upsampling_steps) in range(1, 6)

        if len(upsampling_steps) >= 1:
            self.layers.append(
                nn.Sequential(
                    encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1
                )
            )  # encoder.layer0.conv1
        if len(upsampling_steps) >= 2:
            self.layers.append(nn.Sequential(encoder.pool, encoder.layer1))
        if len(upsampling_steps) >= 3:
            self.layers.append(encoder.layer2)
        if len(upsampling_steps) >= 4:
            self.layers.append(encoder.layer3)
        if len(upsampling_steps) == 5:
            self.layers.append(encoder.layer4)

        layer_channels = [3, 64, 256, 512, 1024, 2048]

        self.feature_info = [
            {
                "in_channels": layer_channels[i],
                "out_channels": layer_channels[i + 1],
                "down_sample_rate": 2,
            }
            for i in range(len(self.layers))
        ]

        # Create conv layers for each hidden state
        self.conv_layers = nn.ModuleList()
        for i, step in enumerate(upsampling_steps):
            self.conv_layers.append(
                nn.Conv2d(
                    self.feature_info[len(upsampling_steps) - i - 1]["out_channels"],
                    step["feature_dim"],
                    1,
                    1,
                )
            )

    """
    x: (batch_size, channels=3, image_size=1024, image_size=1024)
    hidden_states: [
        (bs, 64, 512, 512),
        (bs, 256, 256, 256),
        (bs, 512, 128, 128),
        (bs, 1024, 64, 64),
        (bs, 2048, 32, 32),
    ]
    """

    def forward(self, x):
        feature_map: List[torch.Tensor] = []
        for i, l in enumerate(self.layers):
            x = l.forward(x)
            y = self.conv_layers[len(self.conv_layers) - 1 - i].forward(x)
            feature_map.append(y)
        return feature_map

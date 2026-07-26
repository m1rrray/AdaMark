"""RevealingNet: extracts the watermark/secret from a possibly attacked container

Layer attribute names are part of the checkpoint format and must not be renamed.
"""

import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two dilated convolutions with instance norm and a residual connection"""

    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + x)


class RevealingNet(nn.Module):
    """Recovers the secret payload from a possibly attacked container

    Dilated residual blocks widen the receptive field at constant spatial
    resolution, so the output supports pixel-wise tamper decisions.
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )

        dilation_rates = [2, 2, 2, 2, 4, 4, 4, 4, 1]
        res_blocks = [ResidualBlock(128, d) for d in dilation_rates]
        self.res_blocks = nn.Sequential(*res_blocks)

        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, tampered_container):
        x = self.down_sample(tampered_container)
        x = self.res_blocks(x)
        x = self.up_sample(x)
        retrieved_secret = self.final(x)
        return retrieved_secret

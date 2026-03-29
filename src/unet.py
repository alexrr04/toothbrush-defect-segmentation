import torch as tc
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """Applies Convolution -> BatchNorm -> ReLU twice."""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for image segmentation."""

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.contracting_path = nn.ModuleList()
        self.expansive_path = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path (Left Side)
        for feature in features:
            self.contracting_path.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Expansive path (Right Side)
        for feature in reversed(features):
            self.expansive_path.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.expansive_path.append(DoubleConv(feature * 2, feature))

        # Final layer to map features to the required binary mask output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Contracting path
        for conv in self.contracting_path:
            x = conv(x)
            skip_connections.append(x)  # Save for the expansive path
            x = self.pool(x)

        x = self.bottleneck(x)

        # Reverse the skip connections list so they match the upward steps
        skip_connections = skip_connections[::-1]

        # Expansive Path
        for i in range(0, len(self.expansive_path), 2):
            x = self.expansive_path[i](x)  # Up-convolution step
            skip_connection = skip_connections[i // 2]

            # Handle edge cases where image dimensions aren't perfectly divisible by 16
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concatenate the skip connection from the contracting path
            concat_skip = tc.cat((skip_connection, x), dim=1)

            # Double convolution step
            x = self.expansive_path[i + 1](concat_skip)

        return self.final_conv(x)

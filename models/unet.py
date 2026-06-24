import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive. Got {in_channels}"
            )

        if out_channels <= 0:
            raise ValueError(
                f"out_channels must be positive. Got {out_channels}"
            )

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D tensor. Got shape {x.shape}"
            )

        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            (in_channels // 2) + skip_channels,
            out_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:

        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            raise ValueError(
                f"Spatial dimension mismatch. "
                f"Upsampled tensor: {x.shape}, "
                f"Skip tensor: {skip.shape}"
            )

        x = torch.cat(
            [skip, x],
            dim=1
        )

        x = self.conv(x)

        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive. Got {in_channels}"
            )

        if out_channels <= 0:
            raise ValueError(
                f"out_channels must be positive. Got {out_channels}"
            )

        self.enc1 = DoubleConv(
            in_channels,
            64
        )

        self.enc2 = DownBlock(
            64,
            128
        )

        self.enc3 = DownBlock(
            128,
            256
        )

        self.enc4 = DownBlock(
            256,
            512
        )

        self.bottleneck = DownBlock(
            512,
            1024
        )

        self.up4 = UpBlock(
            1024,
            512,
            512
        )

        self.up3 = UpBlock(
            512,
            256,
            256
        )

        self.up2 = UpBlock(
            256,
            128,
            128
        )

        self.up1 = UpBlock(
            128,
            64,
            64
        )

        self.final_conv = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape "
                f"(B,C,H,W). Got {x.shape}"
            )

        if x.shape[1] != 3:
            raise ValueError(
                f"Expected 3 input channels. "
                f"Got {x.shape[1]}"
            )

        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        skip4 = self.enc4(skip3)

        bottleneck = self.bottleneck(skip4)

        x = self.up4(
            bottleneck,
            skip4
        )

        x = self.up3(
            x,
            skip3
        )

        x = self.up2(
            x,
            skip2
        )

        x = self.up1(
            x,
            skip1
        )

        x = self.final_conv(x)

        return x

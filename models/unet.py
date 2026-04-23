import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Optional, Tuple


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block: Upsample -> Conv -> Concat -> DoubleConv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Identity()
        
        self.double_conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        x1 = self.conv(x1)
        
        # Handle size mismatch (if input size is not divisible by 2^n_downsamples)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)


class UNet_unc(nn.Module):
    """
    UNet with uncertainty estimation for pixel-wise regression.
    
    Architecture:
    - 3 downsampling layers (as requested)
    - 3 upsampling layers
    - Two output heads: mean and variance (log-variance)
    - Dropout for MC Dropout uncertainty estimation
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 1 for height map)
        features: Base number of features (default: 64)
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
        drp_p: Dropout probability for uncertainty estimation
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: int = 64,
        bilinear: bool = True,
        drp_p: float = 0.2,
        **kwargs: Any
    ) -> None:
        super(UNet_unc, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.drop_rate = drp_p
        
        print(f"UNet_unc: in_channels={in_channels}, out_channels={out_channels}, "
              f"features={features}, drp_p={drp_p}")
        
        # Encoder (downsampling path)
        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        
        # Decoder (upsampling path)
        self.up1 = Up(features * 8, features * 4, bilinear)
        self.up2 = Up(features * 4, features * 2, bilinear)
        self.up3 = Up(features * 2, features, bilinear)
        
        # Output heads (mean and variance)
        self.outc_m = nn.Conv2d(features, out_channels, kernel_size=1)
        self.outc_v = nn.Conv2d(features, out_channels, kernel_size=1)
        
        # Initialize output layers
        nn.init.normal_(self.outc_m.weight, 0, 0.01)
        nn.init.constant_(self.outc_m.bias, 0)
        nn.init.normal_(self.outc_v.weight, 0, 0.01)
        nn.init.constant_(self.outc_v.bias, 0)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Encoder with skip connections (no dropout — preserves spatial detail)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck dropout only: applied at the most compressed representation
        # (B, 512, 32, 32) before any decoding, so subsequent spatial convolutions
        # cannot average out the stochastic perturbation. This is the only location
        # where MC Dropout produces meaningful epistemic variance in a fully
        # convolutional network.
        x4 = nn.functional.dropout(x4, p=self.drop_rate, training=True)
        
        # Decoder with skip connections (no dropout — clean skip paths)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Output mean and log-variance
        x_m = self.outc_m(x)
        x_v = self.outc_v(x)
        
        return x_m, x_v


def unet_unc(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet_unc:
    """
    Create UNet with uncertainty estimation.
    
    Note: pretrained parameter is kept for compatibility but not used (no pretrained weights available).
    """
    if pretrained:
        print("Warning: Pretrained weights not available for UNet. Initializing with random weights.")
    
    return UNet_unc(**kwargs)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    def __init__(self, in_channels=512, num_classes=1, skip_channels=[256, 128, 64]):
        """
        in_channels: channels of the deepest feature map (layer4)
        num_classes: number of segmentation classes
        skip_channels: channels from layer3, layer2, layer1 for skip connections
        """
        super().__init__()

        # U-Net decoder: 3 upsampling + conv blocks

        # In the original U-Net, the conv blocks consist of two conv layers each and no batchnorm. 
        # Here we use one conv layer + batchnorm + ReLU for simplicity (our dataset is small :'().

        self.up1 = nn.ConvTranspose2d(in_channels, skip_channels[0], kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(skip_channels[0]*2, skip_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_channels[0]),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(skip_channels[0], skip_channels[1], kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(skip_channels[1]*2, skip_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_channels[1]),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(skip_channels[1], skip_channels[2], kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(skip_channels[2]*2, skip_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_channels[2]),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(skip_channels[2], num_classes, kernel_size=1)


    def forward(self, features, skips=None, out_size=None):
        # Upsample + conv as before
        x = self.up1(features)
        if skips is not None:
            x = torch.cat([x, skips[0]], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        if skips is not None:
            x = torch.cat([x, skips[1]], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        if skips is not None:
            x = torch.cat([x, skips[2]], dim=1)
        x = self.conv3(x)

        # Final conv
        mask = self.final_conv(x)

        # Upsample to target size if provided
        if out_size is not None:
            mask = F.interpolate(mask, size=out_size, mode='bilinear', align_corners=False)
        return mask

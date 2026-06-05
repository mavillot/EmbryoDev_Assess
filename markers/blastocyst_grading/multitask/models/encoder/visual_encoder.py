import torch
import torch.nn as nn
from torchvision.models import resnet34, efficientnet_b3, EfficientNet_B3_Weights



class ResNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        pretrained=True,
        embedding_dim=512
    ):
        super().__init__()

        weights = 'IMAGENET1K_V1' if pretrained else None
        backbone = resnet34(weights=weights)

        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_dim = embedding_dim
        self.skip_channels = [256, 128, 64]

    def forward(self, x):
        """
        x: (B, C, H, W)

        returns:
            features: (B, 512, H/32, W/32) --> segmentation head input
            embedding: (B, 512) --> classification head input
            skips: list of features from layer3, layer2, layer1 for skip connections --> segmentation head input
        """
        x = self.stem(x)
        skip1 = self.layer1(x)
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        features = self.layer4(skip3)

        pooled = self.global_pool(features)
        embedding = pooled.view(pooled.size(0), -1)

        skips = [skip3, skip2, skip1]

        return features, embedding, skips


class EfficientNetB3Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        pretrained=True,
        embedding_dim=512
    ):
        super().__init__()

        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b3(weights=weights)

        if in_channels != 3:
            # Replace the first conv in the stem (features[0][0])
            backbone.features[0][0] = nn.Conv2d(
                in_channels,
                40,  # EfficientNetB3 stem output channels
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            )

        features = backbone.features

        # EfficientNetB3 feature blocks and their output channels:
        # features[0]     -> stem          -> (B, 40,  H/2,  W/2)
        # features[1]     -> MBConv block  -> (B, 24,  H/2,  W/2)   <- skip1
        # features[2]     -> MBConv block  -> (B, 32,  H/4,  W/4)   <- skip2
        # features[3]     -> MBConv block  -> (B, 48,  H/8,  W/8)   <- skip3
        # features[4]     -> MBConv block  -> (B, 96,  H/16, W/16)
        # features[5]     -> MBConv block  -> (B, 136, H/16, W/16)
        # features[6]     -> MBConv block  -> (B, 232, H/32, W/32)
        # features[7]     -> MBConv block  -> (B, 384, H/32, W/32)  <- features
        # features[8]     -> head conv     -> (B, 1536,H/32, W/32)

        self.stem    = features[0]
        self.block1  = features[1]   # H/2
        self.block2  = features[2]   # H/4
        self.block3  = features[3]   # H/8
        self.block4  = nn.Sequential(features[4], features[5])  # H/16
        self.block5  = nn.Sequential(features[6], features[7])  # H/32

        # Project to match ResNet's 512-dim embedding for drop-in compatibility
        self.proj = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU()
        )

        self.global_pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_dim = embedding_dim
        self.skip_channels = [136, 48, 32]
        

    def forward(self, x):
        """
        x: (B, C, H, W)

        returns:
            features:  (B, 512, H/32, W/32)  -- projected to match ResNet interface
            embedding: (B, 512)
            skips:     [skip3 (48ch), skip2 (32ch), skip1 (24ch)]

        Note: skip channel dims differ from ResNet34 (256, 128, 64).
              If your decoder uses fixed channel sizes, add projection
              layers in the decoder or pass skip_channels to it.
        """
        x = self.stem(x)

        x = self.block1(x)        # H/2
        skip1 = self.block2(x)    # H/4

        skip2 = self.block3(skip1)  # H/8

        skip3 = self.block4(skip2)  # H/16

        x = self.block5(skip3)      # H/32
        features = self.proj(x)      # 384ch

        pooled    = self.global_pool(features)
        embedding = pooled.view(pooled.size(0), -1)

        skips = [skip3, skip2, skip1]

        return features, embedding, skips
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_features=512, num_classes=1):
        """
        in_features: size of embedding from encoder (layer4 global pool) / or from temporal transformer
        num_classes: 1 for binary, >1 for multi-class
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

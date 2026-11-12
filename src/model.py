import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class CorruptDetectNet(nn.Module):
    """
    Multi-task MobileNetV3-Small for simultaneous Object Classification (10 classes)
    and Corruption Family Detection (5 classes: Clean, Noise, Blur, Weather, Digital).
    """
    def __init__(self, num_objects: int = 10, num_corruptions: int = 5, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)

        # Engineering optimization: change stem stride from 2 -> 1 for 32x32 edge inputs
        base.features[0][0].stride = (1, 1)

        self.backbone = base.features
        self.pool = base.avgpool
        in_features = 576  # MobileNetV3-Small pooled feature dimension

        # Task A: Object Classification Head
        self.object_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_objects)
        )

        # Task B: Corruption Detection Head (AMD requirement)
        self.corruption_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(128, num_corruptions)
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)
        object_logits = self.object_head(feat)
        corruption_logits = self.corruption_head(feat)
        return object_logits, corruption_logits

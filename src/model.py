import torch.nn as nn
from torchvision import models

class CattleBreedClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes, device):
    model = CattleBreedClassifier(num_classes)
    model = model.to(device)
    return model
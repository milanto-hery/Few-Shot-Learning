import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class PrimatePrototypicalNet(nn.Module):
    def __init__(self):
        super(PrimatePrototypicalNet, self).__init__()
        # Use ResNet18 pre-trained on ImageNet
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the final classification layer to get raw embeddings (512-dim)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

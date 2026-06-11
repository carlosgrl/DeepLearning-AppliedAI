"""
src/models.py — Model definitions for the LMC study.

Architectures chosen to *not* overlap with those in the course notebooks
(which use basic CNNs, VGG, ResNet):

  1. MLP3          — 3-hidden-layer MLP for 28×28 grayscale (KMNIST)
  2. SimpleConvBN  — 5-layer ConvNet with BatchNorm for 32×32 RGB (SVHN)
"""

import torch
import torch.nn as nn


# ═════════════════════════════════════════════════════════════════════════════
# 1.  MLP-3  (KMNIST — 28×28 grayscale)
# ═════════════════════════════════════════════════════════════════════════════

class MLP3(nn.Module):
    """
    Three-hidden-layer fully-connected network.
    """

    def __init__(self, input_dim: int = 784, hidden: int = 256,
                 num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer0 = nn.Linear(input_dim, hidden)
        self.layer1 = nn.Linear(hidden, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, num_classes)
        self.act0 = nn.ReLU()       
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.act0(self.layer0(x))
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        return self.layer3(x)

    # Convenience: list of (weight_name, bias_name) for alignment code
    @staticmethod
    def layer_names():
        """Return ordered list of layer param name prefixes."""
        return ["layer0", "layer1", "layer2", "layer3"]


# ═════════════════════════════════════════════════════════════════════════════
# 2.  SimpleConvBN  (SVHN — 32×32 RGB)
# ═════════════════════════════════════════════════════════════════════════════

class SimpleConvBN(nn.Module):
    """
    5-layer ConvNet with BatchNorm — a custom architecture not present
    in the course notebooks.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        #Convolutional feature extractor
        self.conv0 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        #Classifier head
        self.fc0 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.act(self.bn0(self.conv0(x))))   # → 32×16×16
        x = self.pool(self.act(self.bn1(self.conv1(x))))   # → 64×8×8
        x = self.pool(self.act(self.bn2(self.conv2(x))))   # → 128×4×4
        x = x.flatten(1)                                    # → 2048
        x = self.act(self.bn_fc(self.fc0(x)))               # → 256
        return self.fc1(x)                                   # → C

    @staticmethod
    def layer_names():
        """Return ordered list of layer param name prefixes."""
        return ["conv0", "conv1", "conv2", "fc0", "fc1"]  

    @staticmethod                                          
    def bn_names():
        """Return BN layer names (for REPAIR)."""
        return ["bn0", "bn1", "bn2", "bn_fc"]



# ═════════════════════════════════════════════════════════════════════════════
# 3.  ConvMixer-256  (STL-10 — 96×96 RGB)
#     Trockman & Kolter, "Patches Are All You Need?", 2022
# ═════════════════════════════════════════════════════════════════════════════

class _ConvMixerBlock(nn.Module):
    """Single ConvMixer block: depthwise conv → GELU → pointwise conv → GELU."""

    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same")
        self.bn1 = nn.BatchNorm2d(dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.bn1(self.dw(x)))
        x = x + residual  # residual connection on depthwise
        x = self.act(self.bn2(self.pw(x)))
        return x


class ConvMixer(nn.Module):
    """
    ConvMixer-256/8 — a patch-based architecture with depthwise +
    pointwise convolutions and GELU activations.
    """

    def __init__(self, dim=256, depth=8, kernel_size=5, patch_size=4,
                 num_classes=10, in_channels=3):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.blocks = nn.Sequential(
            *[_ConvMixerBlock(dim, kernel_size) for _ in range(depth)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

    @staticmethod
    def layer_names():
        return ["patch_embed", "blocks", "head"]

    def bn_names(self):                      
        names = ["patch_embed.2"]
        for i in range(len(self.blocks)):    
            names.extend([f"blocks.{i}.bn1", f"blocks.{i}.bn2"])
        return names


# ═════════════════════════════════════════════════════════════════════════════
# 4.  MobileNetV3-small wrapper (Eje F — Task Arithmetic / EuroSAT)
# ═════════════════════════════════════════════════════════════════════════════

def get_mobilenetv3(num_classes=10, pretrained=True):
    """
    Return a MobileNetV3-small with head replaced for num_classes.
    Uses torchvision pretrained weights (ImageNet) but architecture is
    NOT ResNet (inverted residuals + SE blocks).
    """
    import torchvision.models as tvm
    if pretrained:
        model = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT)
    else:
        model = tvm.mobilenet_v3_small(weights=None)
    # Replace classifier head
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, num_classes)
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Factory
# ═════════════════════════════════════════════════════════════════════════════

def get_model(name: str, **kwargs) -> nn.Module:
    """
    Factory to instantiate models by name string.
    """
    registry = {
        "mlp3": MLP3,
        "simpleconvbn": SimpleConvBN,
        "convmixer": ConvMixer,
    }
    key = name.lower().replace("-", "").replace("_", "")
  
    # Handle MobileNetV3 separately (not a class, it's a function)
    if key == "mobilenetv3":
        return get_mobilenetv3(**kwargs)
    
    if key not in registry:
        raise ValueError(f"Unknown model '{name}'. Options: {list(registry)}")
    return registry[key](**kwargs)

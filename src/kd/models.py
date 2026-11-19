"""Teacher (ResNet-50) and student (MobileNetV2, SqueezeNet, EfficientNet-Lite0) networks.

All networks are adapted to 32x32 CIFAR inputs by removing early down-sampling so that every
model produces:
  * a "mid" feature map at 8x8   (used for intermediate feature distillation)
  * a "final" feature map at 4x4 (last conv features, before global pooling)
followed by global-average-pool -> dropout -> linear classifier.

ImageNet-pretrained weights are reused where available; only strides are changed, so all
pretrained kernels still load.

| model              | mid (8x8) channels | final (4x4) channels |
|--------------------|--------------------|----------------------|
| resnet50 (teacher) | 1024 (layer3)      | 2048 (layer4)        |
| mobilenetv2        | 96   (features[13])| 1280                 |
| squeezenet         | 256  (Fire 7)      | 512                  |
| efficientnet_lite0 | 112  (blocks[4])   | 1280                 |
"""
import torch
import torch.nn as nn
from torchvision import models as tvm

TEACHER = "resnet50"
STUDENTS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]

# Peak learning rates (SGD, batch 128). SqueezeNet has no BatchNorm and diverges at high LR.
DEFAULT_LR = {"resnet50": 0.05, "mobilenetv2": 0.05, "squeezenet": 0.01, "efficientnet_lite0": 0.05}


class KDNet(nn.Module):
    """Backbone + classifier head that can also return intermediate feature maps.

    forward(x)                      -> logits             (used for inference / ONNX export)
    forward(x, return_feats=True)   -> logits, {"mid": ..., "final": ...}
    """

    def __init__(self, body: nn.Module, final_channels: int, num_classes: int,
                 mid_module: nn.Module, mid_channels: int, dropout: float = 0.2):
        super().__init__()
        self.body = body
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(final_channels, num_classes)
        self.final_channels = final_channels
        self.mid_channels = mid_channels
        self._capture = False
        self._mid = None
        mid_module.register_forward_hook(self._store_mid)

    def _store_mid(self, module, inputs, output):
        if self._capture:
            self._mid = output

    @property
    def feature_channels(self) -> dict:
        return {"mid": self.mid_channels, "final": self.final_channels}

    def forward(self, x, return_feats: bool = False):
        self._capture = return_feats
        f = self.body(x)
        logits = self.fc(self.dropout(torch.flatten(self.pool(f), 1)))
        if not return_feats:
            return logits
        feats = {"mid": self._mid, "final": f}
        self._mid, self._capture = None, False
        return logits, feats


class _TimmFeatures(nn.Module):
    """Wraps a timm model so calling it returns the last conv feature map."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_features(x)


def resnet50(num_classes=100, pretrained=True) -> KDNet:
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    m.conv1.stride = (1, 1)  # keep the 7x7 kernel so ImageNet weights still apply; drop max-pool
    body = nn.Sequential(m.conv1, m.bn1, m.relu, m.layer1, m.layer2, m.layer3, m.layer4)
    return KDNet(body, 2048, num_classes, mid_module=m.layer3, mid_channels=1024, dropout=0.0)


def mobilenetv2(num_classes=100, pretrained=True) -> KDNet:
    m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
    m.features[0][0].stride = (1, 1)          # stem conv
    m.features[2].conv[1][0].stride = (1, 1)  # first stride-2 depthwise conv
    return KDNet(m.features, 1280, num_classes, mid_module=m.features[13], mid_channels=96, dropout=0.2)


def squeezenet(num_classes=100, pretrained=True) -> KDNet:
    m = tvm.squeezenet1_1(weights=tvm.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None)
    m.features[0].stride = (1, 1)
    m.features[0].padding = (1, 1)
    # The original conv classifier is replaced by GAP + linear (no ReLU on logits, which KD needs).
    return KDNet(m.features, 512, num_classes, mid_module=m.features[7], mid_channels=256, dropout=0.5)


def efficientnet_lite0(num_classes=100, pretrained=True) -> KDNet:
    import timm
    m = timm.create_model("efficientnet_lite0", pretrained=pretrained, num_classes=0)
    m.conv_stem.stride = (1, 1)
    m.blocks[1][0].conv_dw.stride = (1, 1)
    return KDNet(_TimmFeatures(m), m.num_features, num_classes, mid_module=m.blocks[4],
                 mid_channels=112, dropout=0.2)


BUILDERS = {
    "resnet50": resnet50,
    "mobilenetv2": mobilenetv2,
    "squeezenet": squeezenet,
    "efficientnet_lite0": efficientnet_lite0,
}


def build_model(name: str, num_classes: int = 100, pretrained: bool = True) -> KDNet:
    if name not in BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(BUILDERS)}")
    return BUILDERS[name](num_classes=num_classes, pretrained=pretrained)

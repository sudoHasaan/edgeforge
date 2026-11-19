"""Automated architecture generator: turns a sampled config into a PyTorch network.

Network layout:
    stem (3x3 conv, stride 1) -> stage_0 ... stage_{n-1} -> global avg pool -> dropout -> linear

Each stage stacks `depth` blocks of one type; the first block applies the stage stride
(1, 2, 2, 2) and changes the channel count. Blocks with matching in/out shape and stride 1
get a residual connection.

Block types:
    conv   - standard kxk conv + BN + act
    dwsep  - depthwise kxk conv + BN + act -> pointwise 1x1 conv + BN + act (MobileNetV1-style)
    mbconv - 1x1 expand -> depthwise kxk -> 1x1 linear projection (MobileNetV2 inverted residual)
"""
import torch
import torch.nn as nn

ACT_LAYERS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "relu6": lambda: nn.ReLU6(inplace=True),
    "hardswish": lambda: nn.Hardswish(inplace=True),
    "silu": lambda: nn.SiLU(inplace=True),
    "leaky_relu": lambda: nn.LeakyReLU(0.1, inplace=True),
}


def conv_bn(c_in, c_out, k, stride=1, groups=1, act=None):
    layers = [nn.Conv2d(c_in, c_out, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(c_out)]
    if act:
        layers.append(ACT_LAYERS[act]())
    return nn.Sequential(*layers)


class Block(nn.Module):
    def __init__(self, kind, c_in, c_out, kernel, stride, expansion, act):
        super().__init__()
        if kind == "conv":
            self.body = conv_bn(c_in, c_out, kernel, stride, act=act)
        elif kind == "dwsep":
            self.body = nn.Sequential(conv_bn(c_in, c_in, kernel, stride, groups=c_in, act=act),
                                      conv_bn(c_in, c_out, 1, act=act))
        elif kind == "mbconv":
            hidden = c_in * expansion
            layers = [conv_bn(c_in, hidden, 1, act=act)] if expansion != 1 else []
            layers += [conv_bn(hidden, hidden, kernel, stride, groups=hidden, act=act),
                       conv_bn(hidden, c_out, 1, act=None)]
            self.body = nn.Sequential(*layers)
        else:
            raise ValueError(f"unknown block type '{kind}'")
        self.residual = stride == 1 and c_in == c_out

    def forward(self, x):
        out = self.body(x)
        return x + out if self.residual else out


class NASNet(nn.Module):
    def __init__(self, config: dict, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.config = config
        act = config["activation"]
        c = config["stem_channels"]
        self.stem = conv_bn(3, c, 3, 1, act=act)
        stages = []
        for s in config["stages"]:
            blocks = []
            for d in range(s["depth"]):
                blocks.append(Block(s["block"], c, s["channels"], s["kernel"], s["stride"] if d == 0 else 1,
                                    s.get("expansion", 1), act))
                c = s["channels"]
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stages(self.stem(x))
        return self.fc(self.dropout(torch.flatten(self.pool(x), 1)))


def build_network(config: dict, num_classes: int = 10) -> NASNet:
    return NASNet(config, num_classes)


@torch.no_grad()
def count_macs(model: nn.Module, resolution: int) -> int:
    """Multiply-accumulate operations for one image (conv + linear layers)."""
    total = [0]

    def conv_hook(m, i, o):
        total[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def fc_hook(m, i, o):
        total[0] += m.in_features * m.out_features

    hooks = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hooks += [m.register_forward_hook(fc_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    model(torch.zeros(1, 3, resolution, resolution, device=dev))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return total[0]

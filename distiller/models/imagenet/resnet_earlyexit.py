import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from .resnet import DistillerBottleneck
import distiller

__all__ = ['resnet50_earlyexit']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def get_exits_def(num_classes):
    expansion = 1 # models.ResNet.BasicBlock.expansion
    exits_def = [('layer1.2.relu3', nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Conv2d(256, 50, kernel_size=7, stride=2, padding=3, bias=True),
                                                  nn.Conv2d(50, 12, kernel_size=7, stride=2, padding=3, bias=True),
                                                  #nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(12 * expansion, num_classes))),
                                                  #distiller.modules.Print())),
                 ('layer2.3.relu3', nn.Sequential(nn.Conv2d(512, 12, kernel_size=7, stride=2, padding=3, bias=True),
                                                  nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  #distiller.modules.Print()))]
                                                  nn.Linear(12 * expansion, num_classes)))]
    return exits_def


class ResNetEarlyExit(models.ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ee_mgr = distiller.EarlyExitMgr()
        self.ee_mgr.attach_exits(self, get_exits_def(num_classes=1000))

    def forward(self, x):
        self.ee_mgr.delete_exits_outputs(self)
        # Run the input through the network (including exits)
        x = super().forward(x)
        outputs = self.ee_mgr.get_exits_outputs(self) + [x]
        return outputs


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetEarlyExit(block, layers, **kwargs)
    assert not pretrained
    return model


def resnet50_earlyexit(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model, with early exit branches.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', DistillerBottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import BasicBlock


__all__ = ['resnet18_earlyexit', 'resnet34_earlyexit', 'resnet50_earlyexit', 'resnet101_earlyexit', 'resnet152_earlyexit']

# The following URLs represent pretrained models without exits and can't be used
# directly for preloading as the topology is slightly altered to account
# for the exits. We'll be providing publically pretrained models for examples
# with exits in future releases.
model_urls = {
    'resnet18_earlyexit': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34_earlyexit': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50_earlyexit': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101_earlyexit': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152_earlyexit': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResNetEarlyExit(models.ResNet):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetEarlyExit, self).__init__(block, layers, num_classes)

        # Define early exit layers
        self.conv1_exit0 = nn.Conv2d(256, 50, kernel_size=7, stride=2, padding=3, bias=True)
        self.conv2_exit0 = nn.Conv2d(50, 12, kernel_size=7, stride=2, padding=3, bias=True)
        self.conv1_exit1 = nn.Conv2d(512, 12, kernel_size=7, stride=2, padding=3, bias=True)
        self.fc_exit0 = nn.Linear(147 * block.expansion, num_classes)
        self.fc_exit1 = nn.Linear(192 * block.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # Add early exit layers
        exit0 = self.avgpool(x)
        exit0 = self.conv1_exit0(exit0)
        exit0 = self.conv2_exit0(exit0)
        exit0 = self.avgpool(exit0)
        exit0 = exit0.view(exit0.size(0), -1)
        exit0 = self.fc_exit0(exit0)

        x = self.layer2(x)

        # Add early exit layers
        exit1 = self.conv1_exit1(x)
        exit1 = self.avgpool(exit1)
        exit1 = exit1.view(exit1.size(0), -1)
        exit1 = self.fc_exit1(exit1)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # return a list of probabilities
        output = []
        output.append(exit0)
        output.append(exit1)
        output.append(x)
        return output


def resnet18_earlyexit(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEarlyExit(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18_earlyexit']))
    return model


def resnet34_earlyexit(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEarlyExit(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34_earlyexit']))
    return model


def resnet50_earlyexit(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEarlyExit(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50_earlyexit']), strict=False)
    return model


def resnet101_earlyexit(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEarlyExit(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101_earlyexit']))
    return model


def resnet152_earlyexit(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEarlyExit(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152_earlyexit']))
    return model

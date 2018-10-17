import torch.nn as nn
import math
import resnext

__all__ = ['resnext18_earlyexit', 'resnext34_earlyexit', 'resnext50_earlyexit', 'resnext101_earlyexit',
           'resnext152_earlyexit']


class ResNeXtEarlyExit(nn.ResNeXt):

    def __init__(self, block, layers, num_classes=1000, cardinality=32):
        self.inplanes = 64
        super(ResNeXtEarlyExit, self).__init__()

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


def resnext18_earlyexit( **kwargs):
    model = ResNeXtEarlyExit(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnext34_earlyexit(**kwargs):
    model = ResNeXtEarlyExit(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnext50_earlyexit(**kwargs):
    model = ResNeXtEarlyExit(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101_earlyexit(**kwargs):
    model = ResNeXtEarlyExit(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152_earlyexit(**kwargs):
    model = ResNeXtEarlyExit(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

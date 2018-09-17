import torch.nn as nn
import math

__all__ = ['resnext18_earlyexit', 'resnext34_earlyexit', 'resnext50_earlyexit', 'resnext101_earlyexit',
           'resnext152_earlyexit']

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=32):
        super(BasicBlock, self).__init__()
        d = planes*2
        self.conv1 = conv3x3(inplanes, d, stride)
        self.bn1 = nn.BatchNorm2d(d)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(d, d, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(d)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=32):
        super(Bottleneck, self).__init__()
        d = planes*2
        self.conv1 = nn.Conv2d(inplanes, d, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, d, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(d)
        self.conv3 = nn.Conv2d(d, d*2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(d*2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXtEarlyExit(nn.Module):

    def __init__(self, block, layers, num_classes=1000, cardinality=32):
        self.inplanes = 64
        super(ResNeXtEarlyExit, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cardinality)
        self.layer2 = self._make_layer(block, 128, layers[1], cardinality, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cardinality, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cardinality, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Define early exit layers
        self.conv1_exit0 = nn.Conv2d(256, 50, kernel_size=7, stride=2, padding=3, bias=True)
        self.conv2_exit0 = nn.Conv2d(50, 12, kernel_size=7, stride=2, padding=3, bias=True)
        self.conv1_exit1 = nn.Conv2d(512, 12, kernel_size=7, stride=2, padding=3, bias=True)
        self.fc_exit0 = nn.Linear(147 * block.expansion, num_classes)
        self.fc_exit1 = nn.Linear(192 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, cardinality=cardinality))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality=cardinality))

        return nn.Sequential(*layers)

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
    """Constructs a ResNeXt-18 model.
    """
    model = ResNeXtEarlyExit(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnext34_earlyexit(**kwargs):
    """Constructs a ResNeXt-34 model.
    """
    model = ResNeXtEarlyExit(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnext50_earlyexit(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXtEarlyExit(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101_earlyexit(**kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model = ResNeXtEarlyExit(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152_earlyexit(**kwargs):
    """Constructs a ResNeXt-152 model.
    """
    model = ResNeXtEarlyExit(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

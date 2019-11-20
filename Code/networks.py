import math

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

__all__ = [locals()]


def TorchVGG16(**kwargs):
    model = torchvision.models.vgg16_bn()
    model.classifier[6] = nn.Linear(4096, 10)
    return model


class MultiLayerPerceptron(nn.Module):
    """
    Two layer perceptron for MNIST
    """

    def __init__(self, **kwargs):
        """
        Constructor
        """
        super(MultiLayerPerceptron, self).__init__()
        self._layers = kwargs.get('layers', [32])

        layers = [nn.Linear(self._layers[0], self._layers[1])]

        for i in range(1, len(self._layers) - 1):
            layers.append(nn.Linear(self._layers[i], self._layers[i + 1]))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self._layers[0])

        for layer in self.layers:
            x = F.relu(layer(x))

        return x


class TwoLayerPerceptron(nn.Module):
    """
    Two layer perceptron for MNIST
    """

    def __init__(self, **kwargs):
        """
        Constructor
        """
        super(TwoLayerPerceptron, self).__init__()
        hidden_units = kwargs.get('hidden_units', 32)
        self.fc1 = nn.Linear(28 * 28, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class ExampleNet(nn.Module):
    """
    Small testing network
    """

    def __init__(self, **kwargs):
        """
        Constructor
        """
        super(ExampleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Forward pass of the net

        :param x:
        :return:
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def torch_vgg11():
    """
    VGG-11

    :return:
    """
    return models.vgg11(pretrained=False)


def torch_vgg11_bn():
    """
    VGG-11 with Batch Normalization

    :return:
    """
    return models.vgg11_bn(pretrained=False)


def torch_vgg16():
    """
    VGG-16

    :return:
    """
    return models.vgg16(pretrained=False)


def torch_vgg16_bn():
    """
    VGG-16 with Batch Normalization

    :return:
    """
    return models.vgg16_bn(pretrained=False)


'''
Everything below taken from https://github.com/chengyangfu/pytorch-vgg-cifar10
'''

__all__ = [
    'VGG',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19_bn',
    'vgg19',
]


class VGG(nn.Module):
    """
    VGG model
    """

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
        [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")"""

    return VGG(make_layers(cfg['A']))


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""

    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")"""

    return VGG(make_layers(cfg['B']))


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""

    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")"""

    return VGG(make_layers(cfg['D']))


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""

    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")"""

    return VGG(make_layers(cfg['E']))


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""

    return VGG(make_layers(cfg['E'], batch_norm=True))

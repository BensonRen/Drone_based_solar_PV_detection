"""
This implementation is based on:
https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py
"""

# Built-in

# Libs

# Pytorch
import torch
import torch.nn as nn
import torch.nn.init as init
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# Own modules
from network import network_utils


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


fire_cfg = {
    '1_0': [
        [(96, 16, 64, 64), (128, 16, 64, 64), (128, 32, 128, 128)],
        [(256, 32, 128, 128), (256, 48, 192, 192),
         (384, 48, 192, 192), (384, 64, 256, 256)],
        [(512, 64, 256, 256)]
    ],
    '1_1': [
        [(64, 16, 64, 64), (128, 16, 64, 64)],
        [(128, 32, 128, 128), (256, 32, 128, 128)],
        [(256, 48, 192, 192), (384, 48, 192, 192),
         (384, 64, 256, 256), (512, 64, 256, 256)]
    ]
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', strides=(2, 2, 2, 2, 2),
                 inter_features=False, fire_cfg=fire_cfg, n_class=1000):
        super(SqueezeNet, self).__init__()
        self.inter_features = inter_features
        self.n_class = n_class
        self.chans = [a[-1][-1] + a[-1][-2] for a in fire_cfg[version]][::-1]

        if version == '1_0':

            self.layer0 = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7,
                          stride=strides[0], padding=3),
                nn.ReLU(inplace=True)
            )
            self.layer1 = self._make_layer(
                fire_cfg[version][0], stride=strides[1])
            self.layer2 = self._make_layer(
                fire_cfg[version][1], stride=strides[2])
            self.layer3 = self._make_layer(
                fire_cfg[version][2], stride=strides[3])

        elif version == '1_1':

            self.layer0 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3,
                          stride=strides[0], padding=1),
                nn.ReLU(inplace=True)
            )
            self.layer1 = self._make_layer(
                fire_cfg[version][0], stride=strides[1])
            self.layer2 = self._make_layer(
                fire_cfg[version][1], stride=strides[2])
            self.layer3 = self._make_layer(
                fire_cfg[version][2], stride=strides[3])
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))
        
        final_conv = nn.Conv2d(
            512, self.n_class, kernel_size=1, stride=strides[4], dilation=4**(2-strides[4]))
        self.layer4 = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True)
        )
        self.chans = [self.n_class] + self.chans + [self.layer0[0].out_channels]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layer(self, fire_cfg, kernel_size=3, stride=2, ceil_mode=True):
        layers = []
        dilation = 2 // stride
        if stride == 1:
            layers.append(nn.ZeroPad2d(dilation))
        layers.append(nn.MaxPool2d(kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, ceil_mode=ceil_mode))
        for cfg in fire_cfg:
            layers.append(Fire(*cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.inter_features:
            layer0 = self.layer0(x)
            layer1 = self.layer1(layer0)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)

            return layer4, layer3, layer2, layer1, layer0
        else:
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            return x


def _squeezenet(version, pretrained, strides, inter_features, progress, **kwargs):
    model = SqueezeNet(version, strides, inter_features, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        pretrained_state = network_utils.sequential_load(
            model.state_dict(), load_state_dict_from_url(model_urls[arch], progress=progress))
        model.load_state_dict(pretrained_state, strict=False)
    return model


def squeezenet1_0(pretrained=False, strides=(2, 2, 2, 2, 2), inter_features=True,  progress=True, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_0', pretrained, strides, inter_features,  progress, **kwargs)


def squeezenet1_1(pretrained=False, strides=(2, 2, 2, 2, 2), inter_features=True,  progress=True, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_1', pretrained, strides, inter_features,  progress, **kwargs)


if __name__ == '__main__':
    model = squeezenet1_0(False, (2, 2, 2, 1, 1), True)
    from torchsummary import summary
    summary(model, (3, 512, 512), device='cpu')
    

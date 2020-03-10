"""

"""


# Built-in

# Libs

# Own modules
from network.backbones import vggnet, resnet, squeezenet, inception


def models(model_name, pretrained, strides, inter_features):
    if 'vgg' in model_name:
        return getattr(vggnet, model_name)(pretrained, strides, inter_features)
    elif 'resnet' in model_name or 'resnext' in model_name or 'res2net' in model_name:
        return getattr(resnet, model_name)(pretrained, strides, inter_features)
    elif 'squeezenet' in model_name:
        return getattr(squeezenet, model_name)(pretrained, strides, inter_features)
    elif 'inception' in model_name:
        return getattr(inception, model_name)(pretrained, strides, inter_features)
    else:
        raise NotImplementedError(
            'Encoder architecture {} is not supported'.format(model_name))


if __name__ == '__main__':
    import torch
    model = models('resnet50', False, (2, 2, 2, 2, 2), True)
    x = torch.randn((1, 3, 512, 512))
    print(model(x))

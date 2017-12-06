import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from utils import BoxCar, Upsampler
__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet121']))
    return model

def densenet121_racnn_glimpse_extractor(weights=None, num_classes=200, 
        freeze_conv1=False, gap=False, use_gpu=True, up=True):
    if weights is not None:
        base_pretrained = False
    else:
        base_pretrained = True
    model = DenseNet_RACNN_Glimpse_Extractor(num_classes=200,
            base_pretrained=base_pretrained, use_gpu=use_gpu, up=up)
    
    if freeze_conv1:
        for param in model.conv1.parameters():
            param.requires_grad = False
   
    if weights is not None:
        w = torch.load(weights)
        model.load_state_dict(w)
    
    return model

def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet169']))
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet201']))
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet161']))
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=1000, conv_only=False):

        super(DenseNet, self).__init__()
        self.conv_only = conv_only
        if self.conv_only:
            print('[ConvOnly] Using DenseNet as feature extractor')

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        if not self.conv_only:
            out = F.avg_pool2d(out, kernel_size=out.size(2), stride=1)
            out = self.classifier(out.view(out.size(0), -1))
        return out

"""
Predict 3 glimpses g = (p1_x, p1_y, p2_x, p2_y)
Crop img using boxcar func. 
Upsample using bilinear interpolation
FC + softmax

img -> CONV1 -> f -> FC ->   g
 |                          |
 |                          |
 +------------+-------------+
              | (crop using boxcar)
              v

        c1    c2   c3

              | (bilinear upsampling)
              v

        C1    C2   C3

"""
class DenseNet_RACNN_Glimpse_Extractor(nn.Module):
    def __init__(self, num_classes=200, glimpses=2, base_pretrained=True,
            num_channels=3, conv_dim = 9, gap=False, use_gpu=True, up=True):
        """
        glimpse_only: return glimpses of dim (s, g, x.shape)
        base_pretrained: load Imagenet weights for CNNs
        glimpses: number of glimpses
        """
        super(DenseNet_RACNN_Glimpse_Extractor, self).__init__()
        
        self.num_fltrs = 1024 # num filters in output of conv
        self.glimpse_dim = 4 # tl_x, tl_y, h, w btw 0,1
        self.g = glimpses # number of glimpses
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.conv_dim = conv_dim
        self.gap = gap
        self.up = up

        self.conv1 = densenet121(pretrained=base_pretrained, conv_only=True)
        if gap:
            self.glimpse_fc = nn.Linear(self.num_fltrs, self.glimpse_dim * self.g)
        else:
            self.glimpse_fc = nn.Linear(self.num_fltrs * conv_dim * conv_dim, self.glimpse_dim * self.g)
        
        self.cropper = BoxCar(use_gpu=use_gpu)
        # upsample sz self.upsampler = Upsampler(set_zero=True)
        self.upsampler = Upsampler()
        # delete original fc
        del self.conv1.classifier
    
    def convert_bb(self, f, H=299, W=299):
        """
        f: (s, g, 4) tl_x tl_y h w (all btw 0 and 1)
        Network predicts top left corner, width and height
        """

        # scale to img dimensions
        # wo clip
        # f[:, :, 0] = f[:, :, 0]*H
        # f[:, :, 1] = f[:, :, 1]*W
        # f[:, :, 2] = f[:, :, 2]*H
        # f[:, :, 3] = f[:, :, 3]*W
        
        # compute br_x and br_y
        # clip them to H and W resp
        f[:, :, 0] = torch.clamp(f[:, :, 0], min=0, max=98)
        f[:, :, 1] = torch.clamp(f[:, :, 1], min=0, max=98)
        f[:, :, 2] = torch.clamp(f[:, :, 2], min=200, max=298)
        f[:, :, 3] = torch.clamp(f[:, :, 3], min=200, max=298)

        f[:, :, 2] = f[:, :, 0] + f[:, :, 2]
        f[:, :, 3] = f[:, :, 1] + f[:, :, 3]
        f = torch.clamp(f, min = 0, max = 298)

        return f.long()

    def forward(self, x):
        # input shape is: (s, c, H, W)
        s = x.size(0) # num samples
        c = x.size(1) # num channels
        H = x.size(2) # img height
        W = x.size(3) # img width

        f = self.conv1(x) # (s, num_fltrs, d, d)
        
        if self.gap:
            f = F.avg_pool2d(f, kernel_size=f.size(2), stride=1) # GAP  
            f = f.view(s, self.num_fltrs)
        else:
            f = f.view(s, self.num_fltrs * self.conv_dim * self.conv_dim)
        
        f = self.glimpse_fc(f)
        f = f.view(s, self.g, self.glimpse_dim)
        # wo clip f = torch.sigmoid(f) # must be btw 0,1
        f = self.convert_bb(f)
        glimpses = self.cropper(x, f) # (s, g, x.shape)
        # upsample sz glimpses = self.upsampler(x, f)
        if self.up:
            glimpses = self.upsampler(glimpses, f)
        return glimpses

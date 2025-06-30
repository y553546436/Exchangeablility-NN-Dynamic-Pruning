import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.confidences import Confidence, StatsTestConfidence, ThresholdConfidence
from .conv_module import MyConv2dBNRelu, MyConv2dRelu, MyConv2dBNRelu, MyConv2dBNShortcut
from copy import deepcopy

class VGG11CIFAR10(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(VGG11CIFAR10, self).__init__()

        self.conv_block1 = self._make_conv_block(3, 64, 1)
        self.conv_block2 = self._make_conv_block(64, 128, 1)
        self.conv_block3 = self._make_conv_block(128, 256, 2)
        self.conv_block4 = self._make_conv_block(256, 512, 2)
        self.conv_block5 = self._make_conv_block(512, 512, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)

    def _make_conv_block(self, in_channels, out_channels, num_convs):
        layers = []
        # layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        # layers.append(nn.ReLU())
        layers.append(MyConv2dRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        for _ in range(num_convs - 1):
            # layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            # layers.append(nn.BatchNorm2d(out_channels))  # BatchNorm after the convolution
            # layers.append(nn.ReLU())
            layers.append(MyConv2dBNRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
    
    def get_conv_modules_static_prune(self):
        conv_modules = []
        for layer in self.conv_block1:
            if isinstance(layer, MyConv2dRelu) or isinstance(layer, MyConv2dBNRelu):
                conv_modules.append(layer)
        for layer in self.conv_block2:
            if isinstance(layer, MyConv2dRelu) or isinstance(layer, MyConv2dBNRelu):
                conv_modules.append(layer)
        for layer in self.conv_block3:
            if isinstance(layer, MyConv2dRelu) or isinstance(layer, MyConv2dBNRelu):
                conv_modules.append(layer)
        for layer in self.conv_block4:
            if isinstance(layer, MyConv2dRelu) or isinstance(layer, MyConv2dBNRelu):
                conv_modules.append(layer)
        for layer in self.conv_block5:
            if isinstance(layer, MyConv2dRelu) or isinstance(layer, MyConv2dBNRelu):
                conv_modules.append(layer)
        return conv_modules

    def get_conv_modules_dynamic_prune(self):
        return self.get_conv_modules_static_prune()[1:] # first layer not exchangeable

    def gather_flops(self):
        flops = self.flops
        for layer in self.conv_block1:
            if isinstance(layer, MyConv2dRelu) or isinstance(layer, MyConv2dBNRelu):
                flops += layer.gather_flops()
        for layer in self.conv_block2:
            if isinstance(layer, MyConv2dRelu) or isinstance(layer, MyConv2dBNRelu):
                flops += layer.gather_flops()
        for layer in self.conv_block3:
            if isinstance(layer, MyConv2dRelu) or isinstance(layer, MyConv2dBNRelu):
                flops += layer.gather_flops()
        for layer in self.conv_block4:
            if isinstance(layer, MyConv2dRelu) or isinstance(layer, MyConv2dBNRelu):
                flops += layer.gather_flops()
        for layer in self.conv_block5:
            if isinstance(layer, MyConv2dRelu) or isinstance(layer, MyConv2dBNRelu):
                flops += layer.gather_flops()
        return flops
    
    def my_eval(self, confidences, gain_ratio_thresholds=None):
        self.flops = 0 # only count flops not counted sub conv modules
        # If confidences is not an iterator, convert it to an iterator
        confidences = deepcopy(confidences)
        if gain_ratio_thresholds is None:
            gain_ratio_thresholds = [0.1] * len(confidences)
        if not hasattr(confidences, '__next__'):
            confidences = iter(confidences)
        gain_ratio_thresholds = deepcopy(gain_ratio_thresholds)
        if not hasattr(gain_ratio_thresholds, '__next__'):
            gain_ratio_thresholds = iter(gain_ratio_thresholds)
        for i, layer in enumerate(self.get_conv_modules_static_prune()):
            if i == 0:
                layer.my_eval() # first layer is not exchangeable
            else:
                layer.my_eval(next(confidences), next(gain_ratio_thresholds))
        self.forward = self.my_forward

    def my_forward(self, x):
        x = self.conv_block1(x)
        self.flops += x.numel() * 4 # maxpool
        x = self.conv_block2(x)
        self.flops += x.numel() * 4 # maxpool
        x = self.conv_block3(x)
        self.flops += x.numel() * 4 # maxpool
        x = self.conv_block4(x)
        self.flops += x.numel() * 4 # maxpool
        x = self.conv_block5(x)
        self.flops += x.numel() * 4 # maxpool

        self.flops += x.numel() * 2 # avgpool, 2 flops per MAC
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        self.flops += x.numel() * (self.fc.in_features * 2 + 1) # linear with bias
        return x

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
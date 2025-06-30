import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.confidences import Confidence, StatsTestConfidence, ThresholdConfidence
from .conv_module import MyConv2dBNRelu, MyConv2dRelu, MyConv2dBNRelu, MyConv2dBNShortcut
from copy import deepcopy

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(
        #     in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv_bn_relu = MyConv2dBNRelu(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv_bn_shortcut = MyConv2dBNShortcut(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(in_planes, self.expansion*planes,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(self.expansion*planes)
            # )
            self.shortcut = MyConv2dBNShortcut(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def get_conv_modules_static_prune(self):
        conv_modules = [self.conv_bn_relu, self.conv_bn_shortcut]
        if not isinstance(self.shortcut, nn.Sequential):
            conv_modules.append(self.shortcut)
        return conv_modules
    
    def get_conv_modules_dynamic_prune(self):
        conv_modules = [self.conv_bn_relu, self.conv_bn_shortcut]
        # cannot dynamically prune shortcut connection
        return conv_modules

    def gather_flops(self):
        flops = self.flops
        flops += self.conv_bn_relu.flops
        flops += self.conv_bn_shortcut.flops
        if not isinstance(self.shortcut, nn.Sequential):
            flops += self.shortcut.flops
        return flops

    def my_eval(self, confidences):
        self.flops = 0 # only count flops not counted submodules
        self.conv_bn_relu.my_eval(next(confidences))
        self.conv_bn_shortcut.my_eval(next(confidences))
        if not isinstance(self.shortcut, nn.Sequential):
            self.shortcut.my_eval()
        self.forward = self.my_forward

    def my_forward(self, x):
        out = self.conv_bn_relu(x)
        shortcut_vals = self.shortcut(x)
        self.conv_bn_shortcut.set_shortcut_vals(shortcut_vals)
        out = self.conv_bn_shortcut(out)
        out += shortcut_vals
        self.flops += out.numel()
        out = F.relu(out)
        self.flops += out.numel()
        return out

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        # out = F.relu(out)
        out = self.conv_bn_relu(x)
        out = self.conv_bn_shortcut(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 64

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.conv_bn_relu = MyConv2dBNRelu(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def get_conv_modules_static_prune(self):
        conv_modules = [self.conv_bn_relu]
        for layer in self.layer1:
            conv_modules.extend(layer.get_conv_modules_static_prune())
        for layer in self.layer2:
            conv_modules.extend(layer.get_conv_modules_static_prune())
        for layer in self.layer3:
            conv_modules.extend(layer.get_conv_modules_static_prune())
        for layer in self.layer4:
            conv_modules.extend(layer.get_conv_modules_static_prune())
        return conv_modules
    
    def get_conv_modules_dynamic_prune(self):
        conv_modules = [] # cannot dynamically prune first conv_bn_relu
        for layer in self.layer1:
            conv_modules.extend(layer.get_conv_modules_dynamic_prune())
        for layer in self.layer2:
            conv_modules.extend(layer.get_conv_modules_dynamic_prune())
        for layer in self.layer3:
            conv_modules.extend(layer.get_conv_modules_dynamic_prune())
        for layer in self.layer4:
            conv_modules.extend(layer.get_conv_modules_dynamic_prune())
        return conv_modules
            

    def gather_flops(self):
        flops = self.flops
        for layer in self.layer1:
            flops += layer.gather_flops()
        for layer in self.layer2:
            flops += layer.gather_flops()
        for layer in self.layer3:
            flops += layer.gather_flops()
        for layer in self.layer4:
            flops += layer.gather_flops()
        return flops

    def my_eval(self, confidences):
        self.flops = 0 # only count flops not counted sub conv modules
        # If confidences is not an iterator, convert it to an iterator
        confidences = deepcopy(confidences)
        if not hasattr(confidences, '__next__'):
            confidences = iter(confidences)
        for layer in self.layer1:
            layer.my_eval(confidences)
        for layer in self.layer2:
            layer.my_eval(confidences)
        for layer in self.layer3:
            layer.my_eval(confidences)
        for layer in self.layer4:
            layer.my_eval(confidences)
        self.forward = self.my_forward

    # def calculate_flops(self, channel_used_list):
    #     if not hasattr(channel_used_list, '__next__'):
    #         channel_used_list = iter(channel_used_list)
    #     x = torch.randn(1, 3, 32, 32)
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     flops = out.numel() * 3 * 3 * 3
    #     flops += 3 * out.numel() # relu and bn
    #     out, channel_used_list, flops = self.layer1((out, channel_used_list, flops))
    #     out, channel_used_list, flops = self.layer2((out, channel_used_list, flops))
    #     out, channel_used_list, flops = self.layer3((out, channel_used_list, flops))
    #     out, channel_used_list, flops = self.layer4((out, channel_used_list, flops))
    #     out = F.avg_pool2d(out, 4)
    #     flops += out.numel() * 4 * 4
    #     out = out.view(out.size(0), -1)
    #     out = self.linear(out)
    #     flops += out.numel() * self.linear.in_features * 2 # linear with bias
    #     return flops

    def my_forward(self, x):
        out = self.conv_bn_relu(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        self.flops += out.numel() * 4 * 4 * 2 # avgpool, 2 flops per MAC
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        self.flops += out.numel() * (self.linear.in_features * 2 + 1) # linear with bias
        return out

    def forward(self, x):
        out = self.conv_bn_relu(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_CIFAR():
    return ResNet_CIFAR(BasicBlock, [2, 2, 2, 2])
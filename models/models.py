import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import vgg11

from tqdm import tqdm

from .modified_resnet_cifar import ResNet18_CIFAR
from .gnn import GNN
from .modified_vgg_cifar import VGG11CIFAR10
from config import ARTIFACTS_DIR
import os
import urllib.request
import json
import numpy as np

def get_model_param_path(model_name, prune_rate=0):
    if prune_rate > 0:
        return os.path.join(ARTIFACTS_DIR, f"{model_name}-prune-rate-{prune_rate}_params.pth")
    else:
        return os.path.join(ARTIFACTS_DIR, f"{model_name}_params.pth")

def get_iterative_pruned_model_name(model_name, iter, decay_rate):
    return f"{model_name}-{decay_rate}decay-prune-iter-{iter}"

def get_iterative_pruned_model_param_path(model_name, iter, decay_rate):
    remain_rate = 1
    for i in range(iter+1):
        if i == iter:
            prune_rate = 1 - remain_rate
            new_model_name = f"{model_name}-{decay_rate}decay-prune-iter-{i}"
        remain_rate *= decay_rate
    return get_model_param_path(new_model_name, prune_rate)

def save_iterative_pruned_model(model_name, model, iter, decay_rate):
    model_param_path = get_iterative_pruned_model_param_path(model_name, iter, decay_rate)
    torch.save(model.state_dict(), model_param_path)

def load_iterative_pruned_model(model_name, model, iter, decay_rate):
    model_param_path = get_iterative_pruned_model_param_path(model_name, iter, decay_rate)
    model.load_state_dict(torch.load(model_param_path))
    return model

def save_model(model_name, model, prune_rate=0):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_param_path = get_model_param_path(model_name, prune_rate)
    torch.save(model.state_dict(), model_param_path)

def load_model(model_name, model, prune_rate=0):
    model_param_path = get_model_param_path(model_name, prune_rate)
    model.load_state_dict(torch.load(model_param_path))
    return model

def save_model_params(model_name, model_params, prune_rate=0):
    model_param_path = get_model_param_path(model_name, prune_rate)
    torch.save(model_params, model_param_path)

class MNIST_2MLPNet(nn.Module):
    def __init__(self):
        super(MNIST_2MLPNet, self).__init__()
        # First linear layer: input size 28*28 (flattened image), output size 512
        self.fc1 = nn.Linear(28 * 28, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        # Second linear layer: input size 512, output size 10 (number of classes)
        self.fc2 = nn.Linear(512, 10, bias=False)
    
    def forward(self, x):
        # Flatten the input image
        x = x.view(-1, 28 * 28)
        # Apply first layer with ReLU activation
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        # Apply second layer
        x = self.fc2(x)
        return x

def build_mnist_2_mlp_net():
    return "mnist_2_mlp_net", MNIST_2MLPNet()


class MNIST_3MLPNet(nn.Module):
    def __init__(self):
        super(MNIST_3MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10, bias=False)
    
    def forward(self, x):
        # Flatten the input image
        x = x.view(-1, 28 * 28)
        # Apply first layer with ReLU activation
        x = self.relu1(self.fc1(x))
        # Apply second layer
        x = self.relu2(self.fc2(x))
        # Apply third layer
        x = self.fc3(x)
        return x

def build_mnist_3_mlp_net():
    return "mnist_3_mlp_net", MNIST_3MLPNet()

class CIFAR10_2ConvNet(nn.Module):
    def __init__(self):
        super(CIFAR10_2ConvNet, self).__init__()
        # First conv layer: input channels=3, output channels=64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        # Second conv layer: input channels=64, output channels=128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2  = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # Global average pooling reduces the 8x8 feature map to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # Only one FC layer: input channels=128, output classes=10
        self.fc = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        # input size: (batch_size, 3, 32, 32)
        
        x = self.relu1(self.bn1(self.conv1(x))) # (batch_size, 64, 32, 32)
        x = self.pool1(x)                    # (batch_size, 64, 16, 16)
        x = self.relu2(self.bn2(self.conv2(x))) # (batch_size, 128, 16, 16)
        x = self.pool2(x)                    # (batch_size, 128, 8, 8)
        x = self.dropout(x)                 # (batch_size, 128, 8, 8)
        x = self.global_avg_pool(x)         # (batch_size, 128, 1, 1)
        x = x.view(x.size(0), -1)           # (batch_size, 128)
        x = self.fc(x)                      # (batch_size, 10)
        return x


def build_cifar10_2_conv_layer_cnn():
    return "cifar10_2_conv_layer_cnn", CIFAR10_2ConvNet()


class CIFAR10_Simple2ConvNet(nn.Module):
    def __init__(self):
        super(CIFAR10_Simple2ConvNet, self).__init__()
        # First conv layer: input channels=3, output channels=64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        # Second conv layer: input channels=64, output channels=128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        self.pool2  = nn.MaxPool2d(2, 2)
        # Global average pooling reduces the 8x8 feature map to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # Only one FC layer: input channels=128, output classes=10
        self.fc = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        # input size: (batch_size, 3, 32, 32)
        
        x = self.relu1(self.conv1(x)) # (batch_size, 64, 32, 32)
        x = self.pool1(x)                    # (batch_size, 64, 16, 16)
        x = self.relu2(self.conv2(x)) # (batch_size, 128, 16, 16)
        x = self.pool2(x)                    # (batch_size, 128, 8, 8)
        x = self.global_avg_pool(x)         # (batch_size, 128, 1, 1)
        x = x.view(x.size(0), -1)           # (batch_size, 128)
        x = self.fc(x)                      # (batch_size, 10)
        return x


def build_cifar10_simple_2_conv_layer_cnn():
    return "cifar10_simple_2_conv_layer_cnn", CIFAR10_Simple2ConvNet()


class CIFAR10_3ConvNet(nn.Module):
    def __init__(self):
        super(CIFAR10_3ConvNet, self).__init__()
        # First conv layer: input channels=3, output channels=64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False) # feature map size: 32x32
        self.bn1   = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) # feature map size: 16x16
        # Second conv layer: input channels=64, output channels=128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False) # feature map size: 16x16
        self.bn2   = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # feature map size: 8x8
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False) # feature map size: 8x8
        self.bn3   = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.35)
        # Global average pooling reduces the 8x8 feature map to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # Only one FC layer: input channels=128, output classes=10
        self.fc = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        # input size: (batch_size, 3, 32, 32)
        
        x = self.relu1(self.bn1(self.conv1(x))) # (batch_size, 64, 32, 32)
        x = self.pool1(x)                    # (batch_size, 64, 16, 16)
        x = self.relu2(self.bn2(self.conv2(x))) # (batch_size, 128, 16, 16)
        x = self.pool2(x)                    # (batch_size, 128, 8, 8)
        x = self.dropout1(x)                 # (batch_size, 128, 8, 8)
        x = self.relu3(self.bn3(self.conv3(x))) # (batch_size, 256, 8, 8)
        x = self.dropout2(x)                 # (batch_size, 256, 8, 8)
        x = self.global_avg_pool(x)         # (batch_size, 256, 1, 1)
        x = x.view(x.size(0), -1)           # (batch_size, 256)
        x = self.fc(x)                      # (batch_size, 10)
        return x


def build_cifar10_3_conv_layer_cnn():
    return "cifar10_3_conv_layer_cnn", CIFAR10_3ConvNet()


class CIFAR10_Simple3ConvNet(nn.Module):
    def __init__(self):
        super(CIFAR10_Simple3ConvNet, self).__init__()
        # First conv layer: input channels=3, output channels=64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        # Second conv layer: input channels=64, output channels=128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.relu3 = nn.ReLU()
        # Global average pooling reduces the 8x8 feature map to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # Only one FC layer: input channels=128, output classes=10
        self.fc = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        # input size: (batch_size, 3, 32, 32)
        
        x = self.relu1(self.conv1(x)) # (batch_size, 64, 32, 32)
        x = self.pool1(x)                    # (batch_size, 64, 16, 16)
        x = self.relu2(self.conv2(x)) # (batch_size, 128, 16, 16)
        x = self.pool2(x)                    # (batch_size, 128, 8, 8)
        x = self.relu3(self.conv3(x)) # (batch_size, 256, 8, 8)
        x = self.global_avg_pool(x)         # (batch_size, 256, 1, 1)
        x = x.view(x.size(0), -1)           # (batch_size, 256)
        x = self.fc(x)                      # (batch_size, 10)
        return x


def build_cifar10_simple_3_conv_layer_cnn():
    return "cifar10_simple_3_conv_layer_cnn", CIFAR10_Simple3ConvNet()


class CIFAR10_Simple3ConvNet_BN(nn.Module):
    def __init__(self):
        super(CIFAR10_Simple3ConvNet_BN, self).__init__()
        # First conv layer: input channels=3, output channels=64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        # Second conv layer: input channels=64, output channels=128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.relu3 = nn.ReLU()
        # Global average pooling reduces the 8x8 feature map to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.bn3 = nn.BatchNorm2d(256)
        # Only one FC layer: input channels=128, output classes=10
        self.fc = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        # input size: (batch_size, 3, 32, 32)
        
        x = self.relu1(self.conv1(x)) # (batch_size, 64, 32, 32)
        x = self.pool1(x)                    # (batch_size, 64, 16, 16)
        x = self.bn1(x)
        x = self.relu2(self.conv2(x)) # (batch_size, 128, 16, 16)
        x = self.pool2(x)                    # (batch_size, 128, 8, 8)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.relu3(self.conv3(x)) # (batch_size, 256, 8, 8)
        x = self.global_avg_pool(x)         # (batch_size, 256, 1, 1)
        x = self.bn3(x)
        x = x.view(x.size(0), -1)           # (batch_size, 256)
        x = self.fc(x)                      # (batch_size, 10)
        return x


def build_cifar10_simple_3_conv_layer_bn_cnn():
    return "cifar10_simple_3_conv_layer_bn_cnn", CIFAR10_Simple3ConvNet_BN()


class CIFAR10_3ConvGroupNet(nn.Module):
    def __init__(self, group_size):
        self.group_size = group_size
        super(CIFAR10_3ConvGroupNet, self).__init__()
        # First conv layer: input channels=3, output channels=64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        # Second conv layer: input channels=64, output channels=128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, groups=group_size, bias=False)
        self.bn3   = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.35)
        # Global average pooling reduces the 8x8 feature map to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # Only one FC layer: input channels=128, output classes=10
        self.fc = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        # input size: (batch_size, 3, 32, 32)
        
        x = F.relu(self.bn1(self.conv1(x))) # (batch_size, 64, 32, 32)
        x = self.pool(x)                    # (batch_size, 64, 16, 16)
        x = F.relu(self.bn2(self.conv2(x))) # (batch_size, 128, 16, 16)
        x = self.pool(x)                    # (batch_size, 128, 8, 8)
        x = self.dropout1(x)                 # (batch_size, 128, 8, 8)
        x = F.relu(self.bn3(self.conv3(x))) # (batch_size, 256, 8, 8)
        x = self.dropout2(x)                 # (batch_size, 256, 8, 8)
        x = self.global_avg_pool(x)         # (batch_size, 256, 1, 1)
        x = x.view(x.size(0), -1)           # (batch_size, 256)
        x = self.fc(x)                      # (batch_size, 10)
        return x

def build_cifar10_3_conv_group_layer_cnn(group_size):
    return "cifar10_3_conv_group_layer_cnn", CIFAR10_3ConvGroupNet(group_size)


class CIFAR10_3ConvGroupShuffleNet(nn.Module):
    def __init__(self, group_size):
        self.group_size = group_size
        super(CIFAR10_3ConvGroupShuffleNet, self).__init__()
        # First conv layer: input channels=3, output channels=64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        # Second conv layer: input channels=64, output channels=128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=group_size, bias=False)
        self.bn2   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.shuffle = nn.ChannelShuffle(group_size)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, groups=group_size, bias=False)
        self.bn3   = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.35)
        # Global average pooling reduces the 8x8 feature map to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # Only one FC layer: input channels=128, output classes=10
        self.fc = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        # input size: (batch_size, 3, 32, 32)
        
        x = F.relu(self.bn1(self.conv1(x))) # (batch_size, 64, 32, 32)
        x = self.pool(x)                    # (batch_size, 64, 16, 16)
        x = F.relu(self.bn2(self.conv2(x))) # (batch_size, 128, 16, 16)
        x = self.pool(x)                    # (batch_size, 128, 8, 8)
        x = self.dropout1(x)                 # (batch_size, 128, 8, 8)
        x = self.shuffle(x)
        x = F.relu(self.bn3(self.conv3(x))) # (batch_size, 256, 8, 8)
        x = self.dropout2(x)                 # (batch_size, 256, 8, 8)
        x = self.global_avg_pool(x)         # (batch_size, 256, 1, 1)
        x = x.view(x.size(0), -1)           # (batch_size, 256)
        x = self.fc(x)                      # (batch_size, 10)
        return x

def build_cifar10_3_conv_group_shuffle_layer_cnn(group_size):
    return "cifar10_3_conv_group_shuffle_layer_cnn", CIFAR10_3ConvGroupShuffleNet(group_size)

def build_vgg11_bn_cifar10():
    return "cifar10_vgg11_bn", VGG11CIFAR10(num_classes=10, dropout_rate=0.5)

def build_resnet18_cifar10():
    return "cifar10_resnet18", ResNet18_CIFAR()

def build_gnn(num_tasks = 1, num_layer = 5, emb_dim = 300):
    return "gnn", GNN(num_tasks = num_tasks, num_layer = num_layer, emb_dim = emb_dim)

def build_opt():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
    return "opt", model, tokenizer

if __name__ == "__main__":
    vgg11 = build_vgg11_bn_cifar10()
    print(vgg11)
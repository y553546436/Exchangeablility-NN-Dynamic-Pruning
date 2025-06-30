import torch
import torch.nn as nn
from config import DEVICE, early_terminate_it
from utils.confidences import StatsTestConfidence
from copy import deepcopy

# linear layer with early termination, assuming followed directly by ReLU
class MyLinear(nn.Module):
    def __init__(self, linear, confidence):
        super(MyLinear, self).__init__()
        if isinstance(linear, MyLinear):
            linear = linear.linear
        self.linear = linear
        self.confidence = confidence
        self.flops = 0

    def forward(self, x):
        if self.confidence is None:
            y = self.linear(x)
            self.flops += y.numel() * self.linear.weight.shape[1] * 2 # linear layer flops
            return y
        else:
            output_channels, input_channels = self.linear.weight.shape
            y1 = x[:, :early_terminate_it] @ self.linear.weight.data[:, :early_terminate_it].T
            cum_squared = torch.zeros_like(y1)
            for c in range(early_terminate_it):
                tmp = x[:, c:c+1] @ self.linear.weight.data[:, c:c+1].T
                cum_squared += tmp.square_()
            passed_mask = self.confidence(y1, cum_squared, 1, 0, early_terminate_it)
            y2 = x[:, early_terminate_it:] @ self.linear.weight.data[:, early_terminate_it:].T
            y = torch.where(passed_mask, 0, y1+y2)
            self.flops += (passed_mask.sum().item() * early_terminate_it + (~passed_mask).sum().item() * input_channels) * 2 # linear layer flops
            self.flops += passed_mask.numel() * (early_terminate_it * 2 + 6) if isinstance(self.confidence, StatsTestConfidence) else passed_mask.numel()
            return y
    
    def gather_flops(self):
        return self.flops
        
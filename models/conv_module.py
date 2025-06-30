import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE, early_terminate_it
from utils.confidences import StatsTestConfidence, ThresholdConfidence
from static_prune.masks import UnstructuredMask

def get_pruned_conv_flops(output_mask, weight_mask):
    # does not include bias calculation
    channel_output_mask = torch.sum(output_mask.to(torch.float), dim=[0, 2, 3])
    channel_flops = torch.sum(weight_mask, dim=[1, 2, 3]) * 2 # 2 flops per MAC
    return torch.sum(channel_output_mask * channel_flops).item()

def early_terminate_convolution(vals, conv, confidence, followed_bn=None, shortcut_vals=None, prune_mask=None, gain_ratio_threshold=0.1):
    if conv.groups > 1:
        raise NotImplementedError("Early termination for grouped conv is not implemented")

    assert isinstance(confidence, StatsTestConfidence) or isinstance(confidence, ThresholdConfidence), "Only stats test or threshold confidence is supported"

    def extract_sample_weight_element_bias(bn, prev_channels):
        sample_weight = bn.weight[None, :, None, None] / torch.sqrt(bn.running_var[None, :, None, None] + bn.eps)
        sample_element_bias = (- bn.running_mean[None, :, None, None] * sample_weight + bn.bias[None, :, None, None]) / prev_channels
        return sample_weight, sample_element_bias
    
    def get_sum_squared_conv(vals, conv_weight):
        output_channels, input_channels, kernel_height, kernel_width = conv_weight.shape
        c = 0
        cum_squared_vals = F.conv2d(vals[:, c:c+1, :, :], conv_weight[:, c:c+1, :, :], stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups).square_()
        for c in range(1, input_channels):
            cum_squared_vals.add_(F.conv2d(vals[:, c:c+1, :, :], conv_weight[:, c:c+1, :, :], stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups).square_())
        return cum_squared_vals
        # tranformed_weight = conv_weight.permute(1, 0, 2, 3).reshape(input_channels * output_channels, 1, kernel_height, kernel_width)
        # res = F.conv2d(vals, tranformed_weight, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=input_channels)
        # batch_size, _, height, width = res.shape
        # transformed_res = res.reshape(batch_size, input_channels, output_channels, height, width).permute(0, 2, 1, 3, 4)
        # return torch.sum(transformed_res ** 2, dim=2)

    conv_weight = conv.weight.data
    prev_channels = vals.shape[1]
    out_channels = conv_weight.shape[0]
    kernel_slice_number_elements = conv_weight[0][0].numel()
    flops = 0 # count flops for conv, including overhead, not for following layers
    # weight and element_bias that will be applied after conv, coming from conv's bias, bn, or shortcut
    sample_weight = torch.ones((1, out_channels, 1, 1), device=DEVICE)
    sample_element_bias = torch.zeros((1, out_channels, 1, 1), device=DEVICE)
    if conv.bias is not None:
        sample_element_bias += conv.bias[None, :, None, None] / prev_channels
    if followed_bn is not None:
        sample_weight, bn_sample_element_bias = extract_sample_weight_element_bias(followed_bn, prev_channels)
        sample_element_bias = bn_sample_element_bias + sample_element_bias * sample_weight
    if shortcut_vals is not None:
        sample_element_bias = shortcut_vals / prev_channels + sample_element_bias

    if prune_mask is None or (prune_mask.get_mask() == 1).all():
        cum_vals = F.conv2d(vals[:, :early_terminate_it, :, :], conv_weight[:, :early_terminate_it, :, :], stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
        cum_squared_vals = torch.zeros_like(cum_vals, device=DEVICE)
        if isinstance(confidence, StatsTestConfidence):
            # only needed for stats test confidence
            cum_squared_vals = get_sum_squared_conv(vals[:, :early_terminate_it, :, :], conv_weight[:, :early_terminate_it, :, :])
        passed_mask = confidence(cum_vals, cum_squared_vals, sample_weight, sample_element_bias, early_terminate_it)
        # every activation with passed test uses early_terminate_it channels (samples)
        # each sample uses kernel_slice_number_elements MAC, 2 flops per MAC
        flops += passed_mask.sum().item() * early_terminate_it * kernel_slice_number_elements * 2
        tmp_vals = F.conv2d(vals[:, early_terminate_it:prev_channels, :, :], conv_weight[:, early_terminate_it:prev_channels, :, :], stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
        cum_vals = torch.where(passed_mask, (-sample_element_bias / sample_weight) * prev_channels, 
                                cum_vals + tmp_vals)
        flops += (~passed_mask).sum().item() * prev_channels * kernel_slice_number_elements * 2
        # overhead for confidence test
        flops += passed_mask.numel() * (early_terminate_it * 2 + 6) if isinstance(confidence, StatsTestConfidence) else passed_mask.numel()
        del tmp_vals
    else:
        tested_out_channels_mask, weight_before_test_mask, weight_after_test_mask = prune_mask.get_masks_for_early_terminate(gain_ratio_threshold)
        prune_mask = prune_mask.get_mask()
        conv_weight = torch.mul(conv_weight, prune_mask)
        # we skip pruned samples when calculating sample sum and squared sum
        # streaming samples until we have enough samples to test

        # Implementation 1: runs well on cpu, but not on mps
        # non_zero_sample_mask = (prune_mask != 0).any(dim=(2, 3)) # shape: [out_channels, prev_channels]
        # sample_cnt = non_zero_sample_mask[:, :early_terminate_it].sum(dim=1)
        # tested_mask = torch.zeros([prune_mask.shape[0]], dtype=torch.bool, device=DEVICE)
        # passed_mask = torch.zeros_like(cum_vals, dtype=torch.bool, device=DEVICE)
        # for c in range(early_terminate_it, prev_channels):
        #     to_be_tested = (sample_cnt == early_terminate_it) & (~tested_mask) # shape: [out_channels]
        #     if to_be_tested.sum() > 0:
        #         # Only perform confidence test for channels marked by to_be_tested
        #         new_passed_mask = confidence(
        #             cum_vals[:, to_be_tested, :, :],
        #             cum_squared_vals[:, to_be_tested, :, :],
        #             weight[:, to_be_tested, :, :],
        #             element_bias[:, to_be_tested, :, :],
        #             early_terminate_it
        #         )
        #         flops += get_pruned_conv_flops(new_passed_mask, prune_mask[to_be_tested, :c, :, :])
        #         # Create a full mask where only tested channels can be marked as passed
        #         full_test_pass_mask = torch.zeros_like(passed_mask, dtype=torch.bool, device=DEVICE)
        #         # Only update the mask for channels that were tested
        #         full_test_pass_mask[:, to_be_tested, :, :] = new_passed_mask
        #         tested_mask |= to_be_tested
        #         passed_mask |= full_test_pass_mask
        #         del new_passed_mask, full_test_pass_mask
        #     sample_cnt += non_zero_sample_mask[:, c].int()
        #     tmp_vals = F.conv2d(vals[:, c:c+1, :, :], conv_weight[:, c:c+1, :, :], stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
        #     cum_vals = torch.where(passed_mask, (-element_bias / weight) * prev_channels, cum_vals + tmp_vals)
        #     cum_squared_vals += tmp_vals * tmp_vals
        #     del tmp_vals
        # flops += get_pruned_conv_flops(~passed_mask, prune_mask)
        # # overhead for confidence test
        # if isinstance(confidence, StatsTestConfidence):
        #     # Calculate the number of samples used for confidence test (capped at early_terminate_it)
        #     samples_used = torch.min(sample_cnt, torch.full_like(sample_cnt, early_terminate_it, device=DEVICE))
        #     # Each sample contributes 2 flops to the confidence calculation
        #     # Plus 6 additional flops per test for the statistical calculation
        #     flops += (samples_used * 2 + tested_mask * 6).sum().item()
        # else:
        #     flops += tested_mask.numel()

        # Implementation 2:
        if tested_out_channels_mask.any():
            conv_weight_before_test = conv_weight[tested_out_channels_mask, :, :, :] * weight_before_test_mask
            cum_vals_to_be_tested = F.conv2d(vals, conv_weight_before_test, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
            # prepare cum_squared_vals
            if isinstance(confidence, StatsTestConfidence):
                cum_squared_vals = get_sum_squared_conv(vals, conv_weight_before_test)
            else:
                cum_squared_vals = torch.zeros_like(cum_vals_to_be_tested, device=DEVICE)
            passed_mask = confidence(cum_vals_to_be_tested, cum_squared_vals, sample_weight[:, tested_out_channels_mask, :, :], sample_element_bias[:, tested_out_channels_mask, :, :], early_terminate_it)
            tmp_vals = F.conv2d(vals, conv_weight[tested_out_channels_mask, :, :, :] * weight_after_test_mask, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
            cum_vals_to_be_tested = torch.where(passed_mask, (-sample_element_bias[:, tested_out_channels_mask, :, :] / sample_weight[:, tested_out_channels_mask, :, :]) * prev_channels, cum_vals_to_be_tested + tmp_vals)
            if (~tested_out_channels_mask).any():
                cum_vals_others = F.conv2d(vals, conv_weight[~tested_out_channels_mask, :, :, :], stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
                batch_size, _, height, width = cum_vals_to_be_tested.shape
                cum_vals = torch.zeros((batch_size, out_channels, height, width), device=DEVICE)
                cum_vals[:, tested_out_channels_mask, :, :] = cum_vals_to_be_tested
                cum_vals[:, ~tested_out_channels_mask, :, :] = cum_vals_others
                # calculate conv flops
                flops += get_pruned_conv_flops(torch.ones_like(cum_vals_others), prune_mask[~tested_out_channels_mask, :, :, :])
            else:
                cum_vals = cum_vals_to_be_tested
            # calculate test overhead
            if isinstance(confidence, StatsTestConfidence):
                flops += cum_vals_to_be_tested.numel() * (2 * early_terminate_it + 6)
            else:
                flops += cum_vals_to_be_tested.numel()
            # calculate conv flops
            flops += get_pruned_conv_flops(torch.ones_like(passed_mask), weight_before_test_mask)
            flops += get_pruned_conv_flops(~passed_mask, weight_after_test_mask)
        else:
            cum_vals = F.conv2d(vals, conv_weight, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
            flops += get_pruned_conv_flops(torch.ones_like(cum_vals), prune_mask)

    if conv.bias is not None:
        cum_vals = torch.where(passed_mask, (-sample_element_bias / sample_weight) * prev_channels, cum_vals + conv.bias[None, :, None, None].expand_as(cum_vals))
        flops += cum_vals.numel()

    return cum_vals, flops


class MyStandaloneConv2d(nn.Module):
    # no bn and relu before next conv, no early termination
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(MyStandaloneConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask = UnstructuredMask(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def my_eval(self, confidence=None):
        self.flops = 0
        self.forward = self.my_forward
        assert confidence is None, "No early termination for standalone conv"

    def my_forward(self, x):
        self.mask.apply(self.conv)
        res = self.conv(x)
        self.flops += get_pruned_conv_flops(torch.ones_like(res), self.mask.get_mask())
        if self.conv.bias is not None:
            self.flops += res.numel()
        return res

    def gather_flops(self):
        return self.flops
    
    def forward(self, x):
        self.mask.apply(self.conv)
        return self.conv(x)
    

class MyConv2dRelu(nn.Module):
    # conv2d + relu, no bn in between, no shortcut
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(MyConv2dRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask = UnstructuredMask(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.relu = nn.ReLU(inplace=True)

    def my_eval(self, confidence=None, gain_ratio_threshold=0.1):
        self.flops = 0
        self.forward = self.my_forward
        self.confidence = confidence
        self.gain_ratio_threshold = gain_ratio_threshold

    def my_forward(self, x):
        if self.confidence is not None:
            res, flops = early_terminate_convolution(x, self.conv, self.confidence, prune_mask=self.mask, gain_ratio_threshold=self.gain_ratio_threshold)
            self.flops += flops
        else:
            self.mask.apply(self.conv)
            res = self.conv(x)
            self.flops += get_pruned_conv_flops(torch.ones_like(res), self.mask.get_mask())
            if self.conv.bias is not None:
                self.flops += res.numel()
        res = self.relu(res)
        self.flops += res.numel()
        return res

    def gather_flops(self):
        return self.flops

    def forward(self, x):
        self.mask.apply(self.conv)
        return self.relu(self.conv(x))


class MyConv2dBNRelu(nn.Module):
    # conv2d + bn + relu
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(MyConv2dBNRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask = UnstructuredMask(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def my_eval(self, confidence=None, gain_ratio_threshold=0.1):
        self.flops = 0
        self.forward = self.my_forward
        self.confidence = confidence
        self.gain_ratio_threshold = gain_ratio_threshold

    def my_forward(self, x):
        if self.confidence is not None:
            res, flops = early_terminate_convolution(x, self.conv, self.confidence, followed_bn=self.bn, prune_mask=self.mask, gain_ratio_threshold=self.gain_ratio_threshold)
            self.flops += flops
        else:
            self.mask.apply(self.conv)
            res = self.conv(x)
            self.flops += get_pruned_conv_flops(torch.ones_like(res), self.mask.get_mask())
            if self.conv.bias is not None:
                self.flops += res.numel()
        res = self.bn(res)
        self.flops += res.numel() * 3
        res = self.relu(res)
        self.flops += res.numel()
        return res

    def gather_flops(self):
        return self.flops

    def forward(self, x):
        self.mask.apply(self.conv)
        return self.relu(self.bn(self.conv(x)))

class MyConv2dBNShortcut(nn.Module):
    # conv2d + bn, considering shortcut for early termination
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(MyConv2dBNShortcut, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask = UnstructuredMask(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def my_eval(self, confidence=None, gain_ratio_threshold=0.1):
        self.flops = 0
        self.forward = self.my_forward
        self.confidence = confidence
        self.gain_ratio_threshold = gain_ratio_threshold

    def set_shortcut_vals(self, shortcut_vals):
        self.shortcut_vals = shortcut_vals

    def my_forward(self, x):
        if self.confidence is not None:
            res, flops = early_terminate_convolution(x, self.conv, self.confidence, followed_bn=self.bn, shortcut_vals=self.shortcut_vals, prune_mask=self.mask, gain_ratio_threshold=self.gain_ratio_threshold)
            self.flops += flops
        else:
            self.mask.apply(self.conv)
            res = self.conv(x)
            self.flops += get_pruned_conv_flops(torch.ones_like(res), self.mask.get_mask())
            if self.conv.bias is not None:
                self.flops += res.numel()
        res = self.bn(res)
        self.flops += res.numel() * 3
        return res

    def gather_flops(self):
        return self.flops

    def forward(self, x):
        self.mask.apply(self.conv)
        return self.bn(self.conv(x))
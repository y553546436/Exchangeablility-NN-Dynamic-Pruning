import torch
import torch.nn as nn
from config import DEVICE, early_terminate_it

class UnstructuredMask(nn.Module):
    def __init__(self, in_planes, planes, kernel_size, stride, padding, dilation, groups, bias=False):
        super(UnstructuredMask, self).__init__()
        self.mask = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.mask.weight.data = torch.ones(self.mask.weight.size()).to(DEVICE)
        self.mask.weight.requires_grad = False
        self.early_terminate_masks = None

    def get_mask(self):
        return self.mask.weight.data
    
    def get_masks_for_early_terminate(self, gain_ratio_threshold=0.1):
        if self.early_terminate_masks is not None:
            # memoization
            return self.early_terminate_masks
        prune_mask = self.mask.weight.data
        # implementation 1: doing test on any out_channel with enough samples, and do not consider # non-zero elements in each in_channel
        # non_zero_sample_mask = (prune_mask != 0).any(dim=(2, 3))
        # # Calculate cumulative sum of non-zero samples along dimension 1 (prev_channels)
        # cum_non_zero_samples = torch.cumsum(non_zero_sample_mask.to(torch.int), dim=1)

        # # Create a mask where we have exactly early_terminate_it non-zero samples
        # # First find where we have at least early_terminate_it samples
        # has_enough_samples = cum_non_zero_samples >= early_terminate_it
        
        # # Create a mask to identify channels that actually have enough samples
        # out_channels_to_be_tested = has_enough_samples.any(dim=1)
        # test_indices = has_enough_samples[out_channels_to_be_tested].to(torch.int).argmax(dim=1)
        # test_indices += 1

        # # prepare samples before the test
        # weight_before_test_mask = prune_mask[out_channels_to_be_tested, :, :, :]
        # weight_after_test_mask = prune_mask[out_channels_to_be_tested, :, :, :]
        # for out_channel in range(test_indices.shape[0]):
        #     num_prev_channels = test_indices[out_channel]
        #     weight_before_test_mask[out_channel, num_prev_channels:, :, :] = 0
        #     weight_after_test_mask[out_channel, :num_prev_channels, :, :] = 0
        
        # implementation 2: heuristically determine which out_channels are economical to test, and account for # non-zero elements in each in_channel
        out_channels, in_channels, height, width = prune_mask.shape
        out_channels_to_be_tested = torch.zeros(out_channels, dtype=torch.bool, device=DEVICE)
        weight_before_test_masks = []
        weight_after_test_masks = []
        for out_channel in range(out_channels):
            mask = prune_mask[out_channel]
            non_zero_elements = (mask != 0).sum(dim=(1, 2))
            # Get indices of in_channels with non-zero elements
            non_zero_indices = torch.nonzero(non_zero_elements, as_tuple=True)[0]
            
            # If there are non-zero indices, sort them by the number of non-zero elements (ascending)
            if len(non_zero_indices) < early_terminate_it:
                continue
            # Get the non_zero_elements counts for the non-zero indices
            non_zero_counts = non_zero_elements[non_zero_indices]
            
            # Sort the indices by their corresponding counts
            sorted_indices = torch.argsort(non_zero_counts)
            sorted_non_zero_indices = non_zero_indices[sorted_indices]
            
            after_test_non_zero_element_sum = non_zero_elements[sorted_non_zero_indices[early_terminate_it:]].sum()
            if after_test_non_zero_element_sum * gain_ratio_threshold < early_terminate_it:
                continue # heuristic, possible savings may not be worth the cost of testing
            out_channels_to_be_tested[out_channel] = True
            before_test_mask = mask.clone()
            before_test_mask[sorted_non_zero_indices[early_terminate_it:], :, :] = 0
            weight_before_test_masks.append(before_test_mask)
            after_test_mask = mask.clone()
            after_test_mask[sorted_non_zero_indices[:early_terminate_it], :, :] = 0
            weight_after_test_masks.append(after_test_mask)
        if out_channels_to_be_tested.any():
            weight_before_test_masks = torch.stack(weight_before_test_masks, dim=0)
            weight_after_test_masks = torch.stack(weight_after_test_masks, dim=0)
        else:
            weight_before_test_masks = None
            weight_after_test_masks = None
        # print(f"out_channels_to_be_tested: {out_channels_to_be_tested.sum()} / {out_channels}")
        self.early_terminate_masks = (out_channels_to_be_tested, weight_before_test_masks, weight_after_test_masks)
        return self.early_terminate_masks
    
    def update(self, new_mask):
        self.mask.weight.data = new_mask

    def apply(self, conv):
        conv.weight.data = torch.mul(conv.weight, self.mask.weight)
import torch
import torch.nn as nn
import numpy as np
from config import DEVICE

class L1Pruner:
    def __init__(self):
        pass

    def unstructured_prune(self, model, prune_rate=0.50):
        # get all the prunable convolutions
        convs = model.get_conv_modules_static_prune()

        # collate all weights into a single vector so l1-threshold can be calculated
        all_weights = torch.Tensor().to(DEVICE)
        for conv in convs:
            all_weights = torch.cat((all_weights.view(-1), conv.conv.weight.view(-1)))
        abs_weights = torch.abs(all_weights.detach())

        threshold = np.percentile(abs_weights.cpu(), prune_rate * 100)

        # prune anything beneath l1-threshold
        for conv in convs:
            conv.mask.update(
                torch.mul(
                    torch.gt(torch.abs(conv.conv.weight), threshold).float(),
                    conv.mask.mask.weight,
                )
            )

    def prune(self, model, prune_rate):
        self.unstructured_prune(model, prune_rate)
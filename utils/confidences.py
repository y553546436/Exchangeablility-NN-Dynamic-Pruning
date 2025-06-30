import torch
from scipy.stats import norm

class Confidence:
    """
    Abstract base class for negative detection methods.
    Subclasses should implement the __call__ method to determine
    if a neuron's activation is confidently negative based on statistical measures.
    """
    def __call__(self):
        """
        Determines if a neuron's activation is confidently negative based on accumulated statistics.
        
        Returns:
            Boolean tensor: True if the neuron's activation is considered confident, False otherwise
        """
        raise NotImplementedError("Subclasses must implement __call__ method")

class StatsTestConfidence(Confidence):
    """
    Confidence class for statistical tests.
    """
    def __init__(self, alpha):
        self.alpha = alpha
        self.test_stat_bound = norm.ppf(alpha)

    def __call__(self, cum_vals: torch.Tensor, cum_squared_vals, weight, element_bias, n):
        # # Calculate mean in-place
        # cur_mean = cum_vals.div(n)
        # cur_mean.mul_(weight).add_(element_bias)
        # # Calculate variance in-place
        # cur_var = cum_squared_vals.div(n)
        # cur_var.sub_(cur_mean.pow(2))
        # cur_var.mul_(weight.pow(2))
        # # Calculate test statistic in-place
        # test_stats = cur_mean.div_(torch.sqrt_(cur_var.div_(n)))
        # res = test_stats < self.test_stat_bound
        # del cur_mean, cur_var, test_stats
        # return res
        cur_mean = cum_vals / n
        cur_mean = cur_mean * weight + element_bias
        cur_var = cum_squared_vals / n - cur_mean * cur_mean
        cur_var = cur_var * weight * weight
        test_stats = cur_mean / torch.sqrt(cur_var / n)
        res = test_stats < self.test_stat_bound
        del cur_mean, cur_var, test_stats
        return res

class ThresholdConfidence(Confidence):
    """
    Confidence class for thresholding.
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, cum_vals, cum_squared_vals, weight, element_bias, n):
        cur_mean = cum_vals / n
        cur_mean = cur_mean * weight + element_bias
        return cur_mean < self.threshold


# not used for now
class StatsTestKnownSigmaConfidence(Confidence):
    """
    Confidence class for statistical tests with known variance.
    """
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, cum_vals, cum_squared_vals, n, mean_to_compare):
        cur_mean = cum_vals / n
        test_stats = (cur_mean - mean_to_compare) * torch.sqrt(n) / self.sigma
        p_values = torch.special.ndtr(test_stats)
        return p_values < self.alpha
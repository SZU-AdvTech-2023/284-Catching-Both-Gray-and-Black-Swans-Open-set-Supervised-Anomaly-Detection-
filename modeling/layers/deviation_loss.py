import torch
import torch.nn as nn
import torch.nn.functional as F
class DynamicWeightedDeviationLoss(nn.Module):
    def __init__(self, confidence_margin=5.0):
        super(DynamicWeightedDeviationLoss, self).__init__()
        self.confidence_margin = confidence_margin
    def forward(self, y_pred, y_true):
        # Dynamic reference value
        ref = torch.normal(mean=0., std=torch.ones_like(y_pred)).cuda()
        # Deviation calculation
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        # Inlier and outlier losses
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs((self.confidence_margin - dev).clamp_(min=0.))
        # Sample weights
        sample_weights = torch.abs(y_true - 0.5)
        # Weighted deviation loss
        dev_loss = (1 - y_true) * inlier_loss * sample_weights + y_true * outlier_loss * sample_weights
        return torch.mean(dev_loss)


import torch
from torch import nn
import torch.nn.functional as F
from ddpm.classifier.densenet import densenet121
import math

class LatentNetwork(nn.Module):

    def __init__(self,
                 num_channels,
                 sample_duration,
                 sample_size,
                 num_classes = 2):

        super(LatentNetwork, self).__init__()

        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        self.num_channels = num_channels

        self.base_model = densenet121(
                            num_classes = self.num_classes,
                            num_channels = self.num_channels,
                            sample_size = self.sample_size,
                            sample_duration = self.sample_duration)
    def forward(self, x):

        out, _ = self.base_model(x)
        return out, out
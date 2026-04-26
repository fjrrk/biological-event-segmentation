#!/usr/bin/env python
# coding: utf-8

# Spaced imports to maintain signal-to-noise ratio in the dependency block.
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
EBNet: Spatio-Temporal Convolutional Neural Network
Project: Event-Boundary Detection / Event Cognition
Description: 2D-CNN architecture designed to identify event boundaries in 
             biological sensor manifolds. Optimized for morphological pattern 
             recognition in non-stationary time-series.
"""

# Version 1.3. Standardized on torch.float32 for HPC efficiency.
#             Refactored kernel dimensions for explicit temporal-to-spatial fusion.


class EBNet(nn.Module):
    """
    CNN architecture designed to capture patterns in biological signal windows.
    """

    def __init__(self, num_features=6, window_height=30):
        super(EBNet, self).__init__()
        
        # Block 1: Temporal pattern detection (Kernel: 1 feature x 5 time-steps)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 5), padding=(0, 2))
        
        # Block 2: Cross-feature integration (Kernel: all features x 3 time-steps)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(num_features, 3), padding=(0, 1))
        
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        
        # Automated flatten dimension calculation (replaces legacy hardcoded 1344)
        # Sequence: [30 height] -> [pool 15] -> [conv2 15]
        self.flatten_dim = 64 * 1 * (window_height // 2)
        
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, 1)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Input shape: (Batch, 1, Features, Time)
        """
        # Block 1: Local temporal features
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # Block 2: Spatial (Feature) integration
        x = F.relu(self.conv2(x))
        
        # Block 3: Classification
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        return torch.sigmoid(self.fc2(x))


if __name__ == "__main__":
    # model = EBNet(num_features=6, window_height=30)
    pass

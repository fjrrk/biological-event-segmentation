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
             biological manifolds (Gaze/Pupil coordinates). Optimized for 
             low-latency inference on non-stationary time-series.
"""

# Version 1.0.Refactored from EBNet.ipynb and legacy ES_finder scripts.
#             Standardized kernel dimensions for (Features x Time) manifolds.


class EBNet(nn.Module):
    """
    Convolutional Neural Network for biological event segmentation.
    Uses a 2D approach to capture dependencies between sensor channels 
    and temporal dynamics simultaneously.
    """

    def __init__(self, input_channels=1, num_features=3, window_height=30):
        super(EBNet, self).__init__()
        
        # Layer 1: Captures local temporal patterns within sensor channels
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=16, 
            kernel_size=(1, 5), 
            padding=(0, 2)
        )
        
        # Layer 2: Integrates across sensor channels (Spatio-Temporal Fusion)
        self.conv2 = nn.Conv2d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=(num_features, 3), 
            padding=(0, 1)
        )
        
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        
        # Dynamic calculation of linear input size based on window height
        # Post-conv1: Height remains 30
        # Post-pool1: Height becomes 15
        # Post-conv2: Height remains 15 (Features reduced to 1)
        self.flatten_dim = 32 * 1 * (window_height // 2)
        
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Input shape: (Batch, 1, Features, Time)
        """
        # First block: Temporal feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # Second block: Cross-feature integration
        x = F.relu(self.conv2(x))
        
        # Flatten and Classify
        x = x.view(-1, self.flatten_dim)
        x = self.dropout(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        
        return x


if __name__ == "__main__":
    # Example model instantiation
    # model = EBNet(num_features=3, window_height=30)
    # print(model)
    pass

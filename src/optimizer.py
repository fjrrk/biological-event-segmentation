#!/usr/bin/env python
# coding: utf-8

# Though the import formatting is not exactly recommended python style, it's spaced to make it easier for me.
from __future__ import print_function, division
import os
import gc
import logging
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import optuna
from optuna.trial import TrialState

# Custom ingestion logic from the research phase
# from ES_Pupillometry_Processed_Data import preprocessed_pupillometry_data


"""
Automated Hyperparameter Optimization for EBNet
Project: Biological Event Segmentation
Description: Bayesian search for optimal CNN architectures to decode event boundaries.
Environment: Optimized for execution on Rutgers Amarel HPC Cluster
"""

# Version 9. Added linear layer number and size options.
#            Added dropout for all layers except final layer.
#            Added detailed progress logging for long cluster runs.
#            Dropped local Mango paths.

warnings.filterwarnings("ignore")

# Force blocking for easier debugging during trial execution
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

global study, trial, device, epoch_dic

class Net(nn.Module):
    def __init__(self, trial, num_conv_layers, num_filters, num_linear_layers, 
                 linear_layer_size, kernel_size, dropout_linear, dropout_conv):
        """
        Custom CNN Architecture with Dynamic Layer configuration.
        Spaced to track the mathematical geometry of the signal tensors.
        """
        super(Net, self).__init__()
        
        in_size_width = 25  
        in_size_height = 27
        padding_size = 20
        
        # Define the convolutional block
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters[0], 
                                               kernel_size=(kernel_size[0], kernel_size[0]), 
                                               dtype=torch.double, padding=padding_size)])
        self.conv_dropouts = nn.ModuleList([nn.Dropout2d(p=dropout_conv[0])])

        # Dimension tracker to prevent the 'ball of yarn' flatten errors
        def calc_pool_dim(size, kernel, pad):
            res = size - (kernel - 1) + 2*pad - 1
            if res % 2 != 0: res += 1
            return int(res / 2)

        curr_w = calc_pool_dim(in_size_width, kernel_size[0], padding_size)
        curr_h = calc_pool_dim(in_size_height, kernel_size[0], padding_size)

        for i in range(1, num_conv_layers):
            self.convs.append(nn.Conv2d(num_filters[i-1], num_filters[i], 
                                        kernel_size=(kernel_size[i], kernel_size[i]), 
                                        dtype=torch.double, padding=padding_size))
            self.conv_dropouts.append(nn.Dropout2d(p=dropout_conv[i]))
            curr_w = calc_pool_dim(curr_w, kernel_size[i], padding_size)
            curr_h = calc_pool_dim(curr_h, kernel_size[i], padding_size)
        
        self.out_feature = num_filters[num_conv_layers-1] * curr_h * curr_w
        self.linears = nn.ModuleList([nn.Linear(self.out_feature, linear_layer_size[0], dtype=torch.double)])

        for i in range(1, num_linear_layers):
            self.linears.append(nn.Linear(linear_layer_size[i-1], linear_layer_size[i], dtype=torch.double))

        self.lin_dropouts = dropout_linear

        # Initialize weights with He initialization—standard for ReLU paths
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
        for lin in self.linears[:-1]:
            nn.init.kaiming_normal_(lin.weight, nonlinearity='relu')

    def forward(self, x):
        """Forward propagation through the trial architecture."""
        for i in range(len(self.convs)):
            x = F.relu(F.max_pool2d(self.conv_dropouts[i](self.convs[i](x)), 2))
        
        x = torch.flatten(x, 1)
        
        for i in range(len(self.linears)-1):
            x = F.relu(self.linears[i](x))
            x = F.dropout(x, p=self.lin_dropouts[i], training=self.training)
            
        return torch.sigmoid(self.linears[-1](x))

def objective(trial):
    """The math core that Optuna tries to maximize."""
    global device
    
    # Suggestion space for architectural search
    num_conv = trial.suggest_int("num_conv_layers", 1, 5)
    filters = [trial.suggest_int(f"num_filter_{i}", 16, 256, 4) for i in range(num_conv)]
    num_lin = trial.suggest_int("num_linear_layers", 2, 8)
    lin_sizes = [trial.suggest_int(f"num_neurons_{i}", 3000//(i*2+1), 6000//(i*2+1), 230) for i in range(num_lin-1)]
    lin_sizes.append(1)
    
    drop_lin = [trial.suggest_float(f"drop_fc_{i}", 0.1, 0.7) for i in range(num_lin-1)]
    drop_conv = [trial.suggest_float(f"drop_conv_{i}", 0.1, 0.7) for i in range(num_conv)]
    kernels = [trial.suggest_int(f"kernel_{i}", 2, 15//num_conv, 1) for i in range(num_conv)]

    model = Net(trial, num_conv, filters, num_lin, lin_sizes, kernels, drop_lin, drop_conv).to(device)
    
    # Optimizer selection logic
    opt_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, opt_name)(model.parameters(), lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
    
    # Training Loop with Early Pruning
    for epoch in range(50):
        # (Training and validation calls here)
        accuracy = 0.0 # Placeholder for evaluation result
        
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        gc.collect()
        
    return accuracy

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Orchestrating search on: {device}", flush=True)
    
    # Create the Optuna study—maximizing for accuracy
    study = optuna.create_study(direction="maximize", study_name="EBNet_V9_Search")
    # study.optimize(objective, n_trials=20)

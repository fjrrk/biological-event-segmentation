#!/usr/bin/env python
# coding: utf-8

# Spaced imports because condensed blocks give me a headache when searching for libs.
from __future__ import print_function, division
import os
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


"""
Hybrid Feature Selection & CNN Training Pipeline
Project: Biological Event Segmentation
Description: Uses Random Forest Gini importance to prune high-dimensional 
             biological signals before 2D-CNN ingestion.
Environment: Rutgers Amarel HPC Cluster
"""

# Version 10. Added Gini Importance pruning logic.
#             Cleaned up DataCurator class for better ETL modularity.
#             Dropped hardcoded Mango paths.

warnings.filterwarnings("ignore")

global device, curator_count

class DataCurator():
    """
    ETL class to transform messy 1D biological signals into 
    balanced 2D morphological tensors.
    """
    def __init__(self, df, height=30):
        self.df = df
        self.height = height
        self.datalist = []
        self.data = None

    def preprocess_pipeline(self, target_col='stim_onset'):
        """
        Handles the class imbalance nightmare by undersampling null-events 
        to match the rare event boundaries.
        """
        colset = self.df.columns
        tmpdf = self.df.dropna(subset=[target_col])
        
        onset_idxs = tmpdf[tmpdf[target_col] == 1].index.tolist()
        non_onset_idxs = tmpdf[tmpdf[target_col] == 0].index.tolist()
        
        # Balanced sampling logic
        random.seed(52497)
        sampled_non_onsets = random.sample(non_onset_idxs, len(onset_idxs))
        pool = sorted(onset_idxs + sampled_non_onsets)
        
        for i in pool:
            if i >= self.height:
                # Create a 2D 'image' of the signal window
                slice_2d = self.df.loc[i-self.height+1:i, colset[:-1]].values
                label = self.df.loc[i, target_col]
                self.datalist.append(np.vstack((slice_2d, [label]*slice_2d.shape[1])))
        
        if self.datalist:
            self.data = np.stack(self.datalist)
            print(f"ETL Complete: Generated {self.data.shape[0]} balanced tensors.", flush=True)
        return self.data

class EBNet(nn.Module):
    """
    CNN designed to see morphological patterns in pupil/BOLD tensors.
    """
    def __init__(self):
        super(EBNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, dtype=torch.double)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, dtype=torch.double)
        
        # Note: 1344 is the derived flatten dimension based on a 30-step window
        self.fc1 = nn.Linear(1344, 512, dtype=torch.double)
        self.fc2 = nn.Linear(512, 1, dtype=torch.double)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))

def run_hybrid_training(data_path, target_col='stim_onset'):
    """Main orchestration for the Gini + CNN hybrid strategy."""
    bdf = pd.read_csv(data_path, index_col=0)
    
    # Phase 1: Random Forest Gini Importance
    # We prune the signals down to the top 6 most informative features
    print("Executing Phase 1: Identifying high-signal biological features...", flush=True)
    X_rf = bdf.drop(columns=[target_col]).fillna(0)
    y_rf = bdf[target_col].fillna(0)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=52497)
    rf.fit(X_rf, y_rf)
    
    indices = np.argsort(rf.feature_importances_)[::-1]
    selected = [X_rf.columns[i] for i in indices[:6]]
    print(f"Top Features Selected: {selected}", flush=True)
    
    # Phase 2: Train the CNN on the morphological windows
    print("Executing Phase 2: CNN training on optimized input space...", flush=True)
    curator = DataCurator(bdf[selected + [target_col]])
    matrix = curator.preprocess_pipeline(target_col=target_col)
    
    if matrix is not None:
        model = EBNet().to(device)
        # Training orchestration logic follows...
        pass

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # run_hybrid_training("./signal_data.csv")

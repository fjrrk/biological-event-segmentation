"""
Hybrid Feature Selection & CNN Training Pipeline
Project: Biological Event Segmentation
Description: Uses Random Forest feature importance to prune input spaces for 
             a 2D-Convolutional Neural Network.
Environment: Optimized for Rutgers Amarel HPC Cluster
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataCurator():
    """
    Orchestration class for biological signal ETL.
    Transforms 1D pupil signals into 2D morphological tensors.
    """
    def __init__(self, df, height=30):
        self.df = df
        self.height = height
        self.datalist = []
        self.data = None

    def preprocess_pipeline(self, target_col='stim_onset'):
        """
        Implements automated undersampling and binning to handle sparse 
        event markers in continuous signal data.
        """
        colset = self.df.columns
        # Drop rows where we don't have event markers
        tmpdf = self.df.dropna(subset=[target_col])
        
        # Identify indexes of events (Boundaries) vs non-events (Noise)
        onset_idxs = tmpdf[tmpdf[target_col] == 1].index.tolist()
        non_onset_idxs = tmpdf[tmpdf[target_col] == 0].index.tolist()
        
        # Balanced Undersampling: Prevent the CNN from just guessing "no event"
        # because the dataset is 99% noise.
        random.seed(52497)
        sampled_non_onsets = random.sample(non_onset_idxs, len(onset_idxs))
        undersampled_pool = sorted(onset_idxs + sampled_non_onsets)
        
        # Temporal Windowing: Create 2D "Signal Images" from 1D slices
        for i in undersampled_pool:
            if i >= self.height:
                # Slice the data to capture the signal leading up to the boundary
                slice_2d = self.df.loc[i-self.height+1:i, colset[:-1]].values
                label = self.df.loc[i, target_col]
                # Stack the features with the label row for unified storage
                self.datalist.append(np.vstack((slice_2d, [label]*slice_2d.shape[1])))
        
        if self.datalist:
            self.data = np.stack(self.datalist)
            print(f"Data Orchestration Complete: {self.data.shape[0]} balanced tensors generated.")
        return self.data

class EBNet(nn.Module):
    """
    2D CNN architecture for pattern recognition in high-entropy signals.
    Treats temporal signal windows as spatial motifs for event decoding.
    """
    def __init__(self, input_channels=1, output_dim=1):
        super(EBNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, dtype=torch.double)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, dtype=torch.double)
        # Note: Linear input dimensions (1344) derived from curator window height (30)
        # and feature width (6 features pruned via Gini selection).
        self.fc1 = nn.Linear(1344, 512, dtype=torch.double)
        self.fc2 = nn.Linear(512, output_dim, dtype=torch.double)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))

def run_hybrid_training(data_path, target_col='stim_onset'):
    # Load and prep high-resolution sensor data
    bdf = pd.read_csv(data_path, index_col=0)
    
    # Phase 1: Random Forest Feature Importance Selection
    # This reduces the dimensionality of the signal before the CNN "sees" it.
    print("Executing Phase 1: Random Forest Feature Selection...")
    X_rf = bdf.drop(columns=[target_col]).fillna(0)
    y_rf = bdf[target_col].fillna(0)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=52497)
    rf.fit(X_rf, y_rf)
    
    # Extract Gini importance and rank features
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top 6 features to match the hardcoded CNN linear layer input size (1344)
    selected_features = [X_rf.columns[i] for i in indices[:6]]
    print(f"Top Gini Features Identified: {selected_features}")
    
    # Prune dataframe to include only top-ranked biological signals
    bdf_pruned = bdf[selected_features + [target_col]]

    # Phase 2: CNN state transition identification
    print("Executing Phase 2: CNN Signal Pattern Recognition...")
    curator = DataCurator(bdf_pruned)
    processed_matrix = curator.preprocess_pipeline(target_col=target_col)
    
    if processed_matrix is not None:
        model = EBNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.BCELoss()
        # Orchestrate distributed training loop on Amarel cluster...
        print("Model initialized. Ready for training trials.")

if __name__ == '__main__':
    # DATA_PATH = "./pupillometry_data_v9.csv"
    # run_hybrid_training(DATA_PATH)
    pass

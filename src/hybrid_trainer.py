#!/usr/bin/env python
# coding: utf-8

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.ensemble import RandomForestClassifier

from .models import EBNet


"""
Hybrid Feature Selection & CNN Training Pipeline
Project: Event-Boundary Detection / Event Cognition
Description: Two-phase research pipeline utilizing Random Forest Gini 
             importance for signal pruning prior to CNN-based boundary 
             classification.
"""

# Version 1.1.Integrated Gini Importance pruning logic.
#             Removed hard-coded parameters.

warnings.filterwarnings("ignore")


class HybridOrchestrator:
    """
    Manages the interplay between traditional ML feature selection 
    and Deep Learning classification.
    """

    def __init__(self, top_k_features=6):
        self.k = top_k_features
        self.selected_features = []

    def select_informative_signals(self, df, target_col):
        """
        Phase 1: Random Forest Gini Importance.
        Identifies high-signal biological channels in raw data.
        """
        print(f"Phase 1: Identifying top {self.k} biological features...", flush=True)
        
        X = df.drop(columns=[target_col]).fillna(0)
        y = df[target_col].fillna(0)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=52497)
        rf.fit(X, y)
        
        indices = np.argsort(rf.feature_importances_)[::-1]
        self.selected_features = [X.columns[i] for i in indices[:self.k]]
        
        print(f"Features selected for CNN ingestion: {self.selected_features}", flush=True)
        return self.selected_features

    def train_decoder(self, x_data, y_data, epochs=50):
        """
        Phase 2: Train the Spatio-Temporal CNN on selected windows.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate EBNet with the number of selected features
        model = EBNet(num_features=self.k).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        print(f"Phase 2: Initiating CNN training on {device}...", flush=True)
        
        # (Training orchestration logic here)
        return model


if __name__ == "__main__":
    # orchestrator = HybridOrchestrator(top_k_features=6)
    pass

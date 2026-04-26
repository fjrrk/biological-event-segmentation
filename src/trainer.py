#!/usr/bin/env python
# coding: utf-8

# Spaced imports to maintain signal-to-noise ratio in the dependency block.
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from torch.utils.data import DataLoader


"""
EBNet Training and Optimization Orchestrator
Project: Event-Boundary Detection / Event Cognition
Description: Framework for model training and automated hyperparameter 
             optimization using Optuna. Designed for repeatable experiments 
             on imbalanced biological event data.
"""

# Version 1.0. Refactored from Optuna_testing_*.py and legacy training loops.


class ModelTrainer:
    """
    Handles the training loop, validation, and optimization of EBNet.
    """

    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()

    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                total_loss += self.criterion(output, batch_y).item()
                
                predictions = (output > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                
        accuracy = correct / len(dataloader.dataset)
        return total_loss / len(dataloader), accuracy


def objective(trial, train_loader, val_loader, device="cpu"):
    """
    Optuna objective function for Bayesian hyperparameter optimization.
    Refactored from Optuna_testing.py legacy scripts.
    """
    # Hyperparameter search space
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
    
    # Initialize model with trial parameters
    from .models import EBNet
    model = EBNet(window_height=30)
    model.dropout = nn.Dropout(dropout_rate)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Pruned training for optimization speed
    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
        # Validation for Optuna pruning
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        trial.report(avg_val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return avg_val_loss


if __name__ == "__main__":
    # Example optimization start
    # study = optuna.create_study(direction="minimize")
    # study.optimize(lambda t: objective(t, t_loader, v_loader), n_trials=50)
    pass

"""
Automated Hyperparameter Optimization for EBNet
Project: Biological Event Segmentation
Dependencies: torch, optuna, pandas, ES_Pupillometry_Processed_Data
Environment: Optimized for execution on Rutgers Amarel HPC Cluster
"""

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
from ES_Pupillometry_Processed_Data import preprocessed_pupillometry_data

# Suppress warnings and handle CUDA blocking for cluster execution
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Experiment Configuration
STUDY_NAME = "EBNet_Optimization_V9_03"
DATA_PATH = "./RF_extracted_features_modified-labels.csv"

# Global tracking for loss metrics
epoch_dic = {}

class Net(nn.Module):
    def __init__(self, trial, num_conv_layers, num_filters, num_linear_layers, linear_layer_size, 
                 kernel_size, dropout_linear, dropout_conv):
        """
        Custom CNN Architecture with Dynamic Layer configuration for Optuna trials.
        """
        super(Net, self).__init__()
        in_size_width, in_size_height, padding_size = 25, 27, 20
        
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters[0], 
                                               kernel_size=(kernel_size[0], kernel_size[0]), 
                                               dtype=torch.double, padding=padding_size)])
        self.conv_dropouts = nn.ModuleList([nn.Dropout2d(p=dropout_conv[0])])

        # Dynamic dimension calculation for pooling
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

        # Weight Initialization (He Initialization)
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
        for lin in self.linears[:-1]:
            nn.init.kaiming_normal_(lin.weight, nonlinearity='relu')

    def forward(self, x):
        for i in range(len(self.convs)):
            x = F.relu(F.max_pool2d(self.conv_dropouts[i](self.convs[i](x)), 2))
        
        x = torch.flatten(x, 1)
        for i in range(len(self.linears)-1):
            x = F.relu(self.linears[i](x))
            x = F.dropout(x, p=self.lin_dropouts[i], training=self.training)
        return torch.sigmoid(self.linears[-1](x))

def train_epoch(network, optimizer, loss_fn, epoch, train_loader, batch_size):
    network.train()
    running_loss = 0.0
    for batch_i, example in enumerate(train_loader):
        data, target = example[:,:-1,:].unsqueeze(1), example[:,-1,0]
        target = target.type(torch.float)

        optimizer.zero_grad()
        output = network(data.to(device)).type(torch.float).squeeze()
        loss = loss_fn(output.to(device), target.to(device))
        
        running_loss += loss.item() * data.shape[0]
        epoch_dic[epoch].append(running_loss)
        loss.backward()
        optimizer.step()

def evaluate(network, test_loader):
    network.eval()
    correct = 0
    with torch.no_grad():
        for example in test_loader:
            data, target = example[:,:-1,:].unsqueeze(1), example[:,-1,0]
            output = network(data.to(device)).type(torch.float)
            correct += output.eq(target.to(device).data.view_as(output)).sum()
    return correct / len(test_loader.dataset)

def objective(trial):
    # Hyperparameter Suggestion Space
    num_conv = trial.suggest_int("num_conv_layers", 1, 5)
    filters = [trial.suggest_int(f"num_filter_{i}", 16, 256, 4) for i in range(num_conv)]
    num_lin = trial.suggest_int("num_linear_layers", 2, 8)
    lin_sizes = [trial.suggest_int(f"num_neurons_{i}", 3000//(i*2+1), 6000//(i*2+1), 230) for i in range(num_lin-1)]
    lin_sizes.append(1)
    
    drop_lin = [trial.suggest_float(f"drop_fc_{i}", 0.1, 0.7) for i in range(num_lin-1)]
    drop_conv = [trial.suggest_float(f"drop_conv_{i}", 0.1, 0.7) for i in range(num_conv)]
    kernels = [trial.suggest_int(f"kernel_{i}", 2, 15//num_conv, 1) for i in range(num_conv)]

    model = Net(trial, num_conv, filters, num_lin, lin_sizes, kernels, drop_lin, drop_conv).to(device)
    
    opt_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, opt_name)(model.parameters(), lr=lr)
    
    cost_name = trial.suggest_categorical("cost", ['binary_cross_entropy', 'l1_loss', 'mse_loss'])
    cost_fn = getattr(F, cost_name)

    for epoch in range(50): # n_epochs
        epoch_key = f'epoch_{epoch}'
        epoch_dic[epoch_key] = []
        train_epoch(model, optimizer, cost_fn, epoch_key, train_loader, 100)
        accuracy = evaluate(model, test_loader)
        
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        gc.collect()
    return accuracy

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, ratio, train_sampler, test_sampler = preprocessed_pupillometry_data(DATA_PATH)

    train_loader = torch.utils.data.DataLoader(X, batch_size=100, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(X, batch_size=10, sampler=test_sampler)

    # Logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    study = optuna.create_study(direction="maximize", study_name=STUDY_NAME)
    study.optimize(objective, n_trials=20)

    print(f"Best Trial Params: {study.best_trial.params}")
    study.trials_dataframe().to_csv(f'./optuna_results_{STUDY_NAME}.csv', index=False)

#!/usr/bin/env python
# coding: utf-8

# Though the import formatting is not exactly recommended python style, it's spaced to make it easier for me.
from __future__ import print_function, division
import os
import gc
import numpy as np
import pandas as pd
import math
import logging

import optuna
from optuna.trial import TrialState

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ES_Pupillometry_Processed_Data import preprocessed_pupillometry_data


"""
Automated Hyperparameter Optimization for EBNet
Dependencies: torch, optuna, pandas
Notes: Optimized for execution on Rutgers Amarel HPC Cluster
"""

# Version 9. Added linear layer number and size options.
#            Added dropout for all layers except final layer.
#            Added more detail to logs.

import warnings
warnings.filterwarnings("ignore")


# Prepare the data
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

name = "RF_extracted_scaled_w_modified_labels_Optuna-testing-9_03"
X, ratio, train_sampler, test_sampler = preprocessed_pupillometry_data("./RF_extracted_features_modified-labels.csv")

# For testing purposes, we also want to keep a record of loss
epoch_dic = {}

class Net(nn.Module):
    def __init__(self, trial, num_conv_layers, num_filters, num_linear_layers, linear_layer_size, 
                 kernel_size, dropout_linear, dropout_conv):
        """Parameters:
            - trial (optuna.trial._trial.Trial): Optuna trial
            - num_conv_layers (int):             Number of convolutional layers
            - num_filters (list, int):           Number of filters of conv layers
            - num_linear_layers (int):           Number of linear layers
            - linear_layer_size (list, int):     Number of neurons in linear layers
            - kernel_size (list, int):           Length of square kernel side
            - dropout_linear (list, float):      List of dropout ratios for linear layers
            - dropout_conv (list, float):        List of dropout ratios for conv layers
        """
        super(Net, self).__init__()              # Initialize parent class
        in_size_width = 25                       # Input image size (28 pixels)    
        in_size_height = 27
        padding_size = 20
        # Convolution filter size

        # Define the convolutional layers
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters[0], 
                                              kernel_size=(kernel_size[0], kernel_size[0]), 
                                              dtype=torch.double, 
                                              padding=padding_size)])                   # List with the Conv layers
        self.conv_dropouts = nn.ModuleList([nn.Dropout2d(p=dropout_conv[0])])

        out_size_width = in_size_width - (kernel_size[0] - 1)  \
                         + 2*padding_size - 1                                              # Width of the output kernel
        if out_size_width%2!=0:
            out_size_width += 1
        out_size_width = int(out_size_width / 2)                                        # Width after pooling
        
        out_size_height = in_size_height - (kernel_size[0] - 1)  \
                          + 2*padding_size - 1                                             # Height of the output kernel
        if out_size_height%2!=0:
            out_size_height += 1
        out_size_height = int(out_size_height / 2)# + 0.5)                                      # Height after pooling

        for i in range(1, num_conv_layers):
            self.convs.append(nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], 
                                        kernel_size=(kernel_size[i], kernel_size[i]), dtype=torch.double,
                                        padding=padding_size))
            self.conv_dropouts.append(nn.Dropout2d(p=dropout_conv[i]))
            out_size_width = out_size_width - (kernel_size[i] - 1) \
                             + 2*padding_size - 1                                          # Width of the output kernel
            if out_size_width%2!=0:
                out_size_width += 1
            out_size_width = int(out_size_width / 2)                                    # Width after pooling

            out_size_height = out_size_height - (kernel_size[i] - 1) \
                              + 2*padding_size - 1                                         # Height of the output kernel
            if out_size_height%2!=0:
                out_size_height += 1
            out_size_height = int(out_size_height / 2)                             # Height after pooling
        
        self.out_feature = num_filters[num_conv_layers-1] * out_size_height * out_size_width  # Size of flattened features
        

        self.linears = nn.ModuleList([nn.Linear(self.out_feature, linear_layer_size[0], dtype=torch.double)])

        for i in range(1, num_linear_layers):
            self.linears.append(nn.Linear(linear_layer_size[i-1], linear_layer_size[i], dtype=torch.double))

        self.lin_dropouts = dropout_linear

        # Initialize weights with the He initialization
        for i in range(1, num_conv_layers):
            nn.init.kaiming_normal_(self.convs[i].weight, nonlinearity='relu')
            if self.convs[i].bias is not None:
                nn.init.constant_(self.convs[i].bias, 0)

        for i in range(1, num_linear_layers):
            nn.init.kaiming_normal_(self.linears[i].weight, nonlinearity='relu')
        nn.init.constant_(self.linears[-1].bias.data, np.log(ratio))

    def forward(self, x):
        """Forward propagation.
        Parameters:
            - x (torch.Tensor): Input tensor of size [N,1,27,25]
        Returns:
            - (torch.Tensor): The output tensor after forward propagation [N,1]
        """
        for i in range(len(self.convs)):  # For each convolutional layer
            x = F.relu(F.max_pool2d(self.conv_dropouts[i](self.convs[i](x)), 2))
        
        x = torch.flatten(x, 1)                   # Flatten tensor
        
        for i in range(len(self.linears)-1):
            x = F.relu(self.linears[i](x))
            x = F.dropout(x, p=self.lin_dropouts[i], training = self.training)
                   
        x = F.sigmoid(self.linears[-1](x))
        return x


def train(network, optimizer, lossfx, epoch):
    """Trains the model.
    Parameters:
        - network (__main__.Net):              The CNN
        - optimizer (torch.optim.<optimizer>): The optimizer for the CNN
    """
    network.train()  # Set the module in training mode (only affects certain modules)
    running_loss = 0.0

    for batch_i, example in enumerate(train_loader):  # For each batch
        data, target = example[:,:-1,:].unsqueeze(1), example[:,-1,0]#.unsqueeze(0)
        target = target.type(torch.float)               # Takes care of ["nll_loss_forward_reduce_cuda_kernel_2d_index" 

        # Limit training data for faster computation
        if batch_i * batch_size_train > number_of_train_examples:
            break

        optimizer.zero_grad()                                 # Clear gradients
        output = network(data.to(device))                     # Forward propagation
        output = output.type(torch.float).squeeze()                # Cast to Long to prevent any 
        loss = lossfx(output.to(device), target.to(device))          # Compute loss (negative log likelihood: −log(y))
        running_loss += loss.item() * data.shape[0]
        epoch_dic[epoch].append(running_loss)
        loss.backward()                                       # Compute gradients
        optimizer.step()                                      # Update weights

def test(network):
    """Tests the model.
    Parameters:
        - network (__main__.Net): The CNN
    Returns:
        - accuracy_test (torch.Tensor): The test accuracy
    """
    network.eval()         # Set the module in evaluation mode (only affects certain modules)
    correct = 0
    with torch.no_grad():  # Disable gradient calculation (when you are sure that you will not call Tensor.backward())
        for batch_i, example in enumerate(test_loader):  # For each batch
            data, target = example[:,:-1,:].unsqueeze(1), example[:,-1,0]
            target = target.type(torch.float)               # Takes care of ["nll_loss_forward_reduce_cuda_kernel_2d_index" 
                                                             # not implemented for 'Double'] error
            # Limit testing data for faster computation
            if batch_i * batch_size_test > number_of_test_examples:
                break

            output = network(data.to(device))               # Forward propagation
            output = output.type(torch.float)
            pred = output.data                              # Find max value in each row, return indexes of max values
            
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()  # Compute correct predictions

    accuracy_test = correct / len(test_loader.dataset)

    return accuracy_test

def objective(trial):
    """Objective function to be optimized by Optuna.
    Hyperparameters chosen to be optimized: optimizer, learning rate,
    dropout values, number of convolutional layers, number of filters of
    convolutional layers, number of neurons of fully connected layers.
    Inputs:
        - trial (optuna.trial._trial.Trial): Optuna trial
    Returns:
        - accuracy(torch.Tensor): The test accuracy. Parameter to be maximized.
    """

    # Define range of values to be tested for the hyperparameters
    max_conv = 5
    max_fc = 8
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, max_conv)   # Number of convolutional layers
    num_filters = [trial.suggest_int("num_filter_"+str(i), 16, 256, 4)
                   for i in range(num_conv_layers)]                # Number of filters for the convolutional layers
    num_linear_layers = trial.suggest_int("num_linear_layers", 2, max_fc) # Number of linear layers
    linear_layer_size = [trial.suggest_int("num_neurons_"+str(i), 3000//(i*2+1), 6000//(i*2+1), 230)
                   for i in range(num_linear_layers-1)]              # Number of neurons for each linear layer
    linear_layer_size.append(trial.suggest_int("num_neurons_"+str(num_linear_layers-1),1 , 1))
    dropout_linear = [trial.suggest_float("drop_fc_"+str(i), 0.1, 0.7)
                   for i in range(num_linear_layers-1)]  
    dropout_conv = [trial.suggest_float("drop_conv_"+str(i), 0.1, 0.7)
                   for i in range(num_conv_layers)]               # Dropout for convolutional layers
    kernel_size = [trial.suggest_int("kernel_"+str(i), 2, max_conv*3//num_conv_layers, 1)
                   for i in range(num_conv_layers)]    

    # Generate the model
    model = Net(trial, num_conv_layers, num_filters, num_linear_layers, linear_layer_size, kernel_size, 
                dropout_linear, dropout_conv).to(device)

    # Generate the optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)                                 # Learning rates
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Generate cost functions
    cost_function = trial.suggest_categorical("cost", ['binary_cross_entropy', 'l1_loss', 'soft_margin_loss', 'mse_loss'])
    cost = getattr(F, cost_function)

    # Training of the model
    for epoch in range(n_epochs):
        epoch_dic['epoch_'+str(epoch)] = []
        train(model, optimizer, cost, 'epoch_'+str(epoch))  # Train the model
        accuracy = test(model)   # Evaluate the model

        # For pruning (stops trial early if not promising)
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        gc.collect()
    return accuracy


if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Optimization study for a PyTorch CNN with Optuna
    # -------------------------------------------------------------------------

    # Use cuda if available for faster computations
    device = torch.device("cuda")

    # --- Parameters ----------------------------------------------------------
    n_epochs = 50                         # Number of training epochs
    batch_size_train = 100                 # Batch size for training data
    batch_size_test = 10                   # Batch size for testing data
    number_of_trials = 20                  # Number of Optuna trials
    limit_obs = True                       # Limit number of observations for faster computation

    # *** Note: For more accurate results, do not limit the observations.
    #           If not limited, however, it might take a very long time to run.
    #           Another option is to limit the number of epochs. ***

    if limit_obs:  # Limit number of observations
        number_of_train_examples = 500 * batch_size_train  # Max train observations
        number_of_test_examples = 50 * batch_size_test      # Max test observations
    else:
        number_of_train_examples = 60000                   # Max train observations
        number_of_test_examples = 10000                    # Max test observations
    # -------------------------------------------------------------------------

    # Make runs repeatable
    random_seed = 123
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)

    # Create data loaders for train and test sets
    train_loader = torch.utils.data.DataLoader(X, batch_size=batch_size_train, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(X, batch_size=batch_size_test, sampler=test_sampler)

    # Start logger
    optuna.logging.disable_default_handler()  # Disable the default handler.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler("./optuna_res/Log_"+name+".log", mode="w"))
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.

    # Create an Optuna study to maximize test accuracy
    study = optuna.create_study(direction="maximize", study_name=name)
    logger.info("\nOptimization Log") 
    logger.info("Final layer activation function: sigmoid")
    logger.info("Random seed value: %s" %str(random_seed))
    logger.info("Batch size for training: %s" % str(batch_size_train))
    logger.info("Batch size for validation: %s" % str(batch_size_test))
    logger.info("Number of epochs: %s" % str(n_epochs))
    logger.info("Number of trials: %s\n" % str(number_of_trials))
    study.optimize(objective, n_trials=number_of_trials)

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy

    df.to_csv('./optuna_res/optuna_results_'+name+'.csv', index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(study, target=None)

    # Display the most important hyperparameters
    print('\nMost important hyperparameters:')
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))

    lossdf = pd.DataFrame.from_dict(epoch_dic)
    lossdf.to_csv('./optuna_res/loss_'+name+'.csv', ignore_index=True)

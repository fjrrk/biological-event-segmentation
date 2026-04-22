#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Though the import formatting is not exactly recommended python style, it's spaced to make it easier for me.
from __future__ import print_function, division
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


# In[ ]:
# Version 8. Added loss function options.
#            Added He initialization for all linear layers.
#            Added more detail to logs.

# import warnings
# warnings.filterwarnings("ignore")


# Prepare the data

name = "RF_extracted_scaled_w_modified_labels_Optuna-testing-8_03"
X, ratio, train_sampler, test_sampler = preprocessed_pupillometry_data("./RF_extracted_features_modified-labels.csv")

# For testing purposes, we also want to keep a record of loss
epoch_dic = {}

class Net(nn.Module):

    def __init__(self, trial, num_conv_layers, num_filters, num_neurons1, num_neurons2, 
                 kernel_size, drop_conv2, drop_fc1, drop_fc2):
        """Parameters:
            - trial (optuna.trial._trial.Trial): Optuna trial
            - num_conv_layers (int):             Number of convolutional layers
            - num_filters (list):                Number of filters of conv layers
            - num_neurons1/2 (int):              Number of neurons of FC layers
            - kernel_size (int):                 Length of square kernel side
            - drop_conv2 (float):                Dropout ratio for conv layer 2
            - drop_fc1 (float):                  Dropout ratio for FC1
        """
        super(Net, self).__init__()                                                     # Initialize parent class
        in_size_width = 25                                                              # Input image size (28 pixels)    
        in_size_height = 27
        padding_size = 20
        # Convolution filter size

        # Define the convolutional layers
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters[0], 
                                              kernel_size=(kernel_size, kernel_size), 
                                              dtype=torch.double, 
                                              padding=padding_size)])                   # List with the Conv layers
        
        out_size_width = in_size_width - (kernel_size - 1)  \
                         + 2*padding_size - 1                                              # Width of the output kernel
        if out_size_width%2!=0:
            out_size_width += 1
        out_size_width = int(out_size_width / 2)                                        # Width after pooling
        
        out_size_height = in_size_height - (kernel_size - 1)  \
                          + 2*padding_size - 1                                             # Height of the output kernel
        if out_size_height%2!=0:
            out_size_height += 1
        out_size_height = int(out_size_height / 2)# + 0.5)                                      # Height after pooling
        # print("Prev | Height: %s, Width: %s" % (str(out_size_height), str(out_size_width)))
        for i in range(1, num_conv_layers):
            self.convs.append(nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], 
                                        kernel_size=(kernel_size, kernel_size), dtype=torch.double,
                                        padding=padding_size))
            out_size_width = out_size_width - (kernel_size - 1) \
                             + 2*padding_size - 1                                          # Width of the output kernel
            if out_size_width%2!=0:
                out_size_width += 1
            out_size_width = int(out_size_width / 2)                                    # Width after pooling

            out_size_height = out_size_height - (kernel_size - 1) \
                              + 2*padding_size - 1                                         # Height of the output kernel
            if out_size_height%2!=0:
                out_size_height += 1
            out_size_height = int(out_size_height / 2)# + 0.5)                                 # Height after pooling
            # print("Post | Height: %s, Width: %s" % (str(out_size_height), str(out_size_width)))

        self.conv2_drop = nn.Dropout2d(p=drop_conv2)                                    # Dropout for conv2
        
        self.out_feature = num_filters[num_conv_layers-1] * out_size_height * out_size_width  # Size of flattened features
        self.fc1 = nn.Linear(self.out_feature, num_neurons1, dtype=torch.double)        # Fully Connected layer 1
        self.fc2 = nn.Linear(num_neurons1, num_neurons2, dtype=torch.double)            # Fully Connected layer 1
        self.fc3 = nn.Linear(num_neurons2, 1, dtype=torch.double)                       # Fully Connected layer 2
        self.p1 = drop_fc1                                                              # Dropout ratio for FC1
        self.p2 = drop_fc2

        # Initialize weights with the He initialization
        for i in range(1, num_conv_layers):
            nn.init.kaiming_normal_(self.convs[i].weight, nonlinearity='relu')
            if self.convs[i].bias is not None:
                nn.init.constant_(self.convs[i].bias, 0)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.constant_(self.fc3.bias.data, np.log(ratio))

    def forward(self, x):
        """Forward propagation.
        Parameters:
            - x (torch.Tensor): Input tensor of size [N,1,27,25]
        Returns:
            - (torch.Tensor): The output tensor after forward propagation [N,1]
        """
        for i, conv_i in enumerate(self.convs):  # For each convolutional layer
            if i > 1:  # Add dropout if layer 2
                x = F.relu(F.max_pool2d(self.conv2_drop(conv_i(x)), 2))  # Conv_i, dropout, max-pooling, RelU
            else:
                x = F.relu(F.max_pool2d(conv_i(x), 2))                   # Conv_i, max-pooling, RelU
            # print("Convolution shape: ", x.shape)
        x = torch.flatten(x, 1)                              # Flatten tensor
        # print("Flattened shape: ", x.shape)
        x = F.relu(self.fc1(x))                              # FC1, RelU
        x = F.dropout(x, p=self.p1, training=self.training)  # Apply dropout after FC1 only when training
        x = F.relu(self.fc2(x))                              # FC2, RelU
        x = F.dropout(x, p=self.p2, training=self.training)  # Apply dropout after FC1 only when training
        x = self.fc3(x)                                      # FC3
        return F.log_softmax(x, dim=1)                       # log(softmax(x))


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
                                                             # not implemented for 'Double'] error

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
            pred = output.data#.max(1, keepdim=True)[1]      # Find max value in each row, return indexes of max values
            
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
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, max_conv)   # Number of convolutional layers
    num_filters = [trial.suggest_int("num_filter_"+str(i), 2, 80, 1)
                   for i in range(num_conv_layers)]                # Number of filters for the convolutional layers

    num_neurons1 = trial.suggest_int("num_neurons1", 76, 4000)  # Number of neurons of FC1 layer
    num_neurons2 = trial.suggest_int("num_neurons2", 38, 2000)  # Number of neurons of FC1 layer
    kernel_size = trial.suggest_int("kernel_size", 2, max_conv*3//num_conv_layers, 1)  # Number of neurons of FC1 layer
    drop_conv2 = trial.suggest_float("drop_conv2", 0.2, 0.5)       # Dropout for convolutional layer 2
    drop_fc1 = trial.suggest_float("drop_fc1", 0.1, 0.5)           # Dropout for FC1 layer
    drop_fc2 = trial.suggest_float("drop_fc2", 0.1, 0.5)           # Dropout for FC1 layer

    # Generate the model
    model = Net(trial, num_conv_layers, num_filters, num_neurons1, num_neurons2, kernel_size, 
                drop_conv2, drop_fc1, drop_fc2).to(device)

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
    n_epochs = 100                         # Number of training epochs
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

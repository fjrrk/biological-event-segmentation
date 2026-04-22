#!/usr/bin/env python
# coding: utf-8

import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import formatrix

def pupillometry_dataset(csv_file):
    data = formatrix.DataCurator(csv_file)
    binmaker = formatrix.BinData(data)
    binned_data = binmaker(25)
    data = formatrix.Undersample(binned_data)
    in_matrix = data()
    return in_matrix


def ratio_check(a_matrix):
    # Check the ratio of positive and negative examples
    # in a matrix with shape batchxfeaturesxexamples
    pos = 0
    neg = 0
    for i in range(a_matrix.shape[0]):
        if a_matrix[i,-1,0]>0:
            pos += 1
        else:
            neg +=1
    return pos/neg


def training_splitter(a_matrix):
    # Here, we split the data into training/testing and shuffle
    random_seed = 324
    breakdown = 0.3
    split = int(len(a_matrix) * breakdown)
    indices = [*range(len(a_matrix))]
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    return train_sampler, test_sampler

def preprocessed_pupillometry_data(csv_file):

    X = pupillometry_dataset(csv_file)
    ratio = ratio_check(X)
    train, test = training_splitter(X)
    X = formatrix.TorchedBins(X)
    return X, ratio, train, test

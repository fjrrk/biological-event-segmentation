#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
import warnings
warnings.filterwarnings("ignore")

class DataCurator():

    def __init__(self, df):
        self.seed = random.seed(52497)
        self.df = df
        self.height = 30
        self.width = 0
        self.end_point = 0
        self.n_pos = None
        self.data = None
        self.undersampled = None
        self.binned = None
        self.datalist = []
        self.past_heights = []

    def get_width(self):
        return self.width

    def get_col_names(self):
        return self.datalist

    def get_binned_data():
        return self.binned

    def set_height(self, n):
        self.height = n
        self.past_heights.append(self.height)
        
    def choose_height(self):
        self.height = random.randint(30, 300)
        self.past_heights.append(self.height)

    def set_labels(self):
        self.data = self.data.loc[:,]

    def shuffle_mat(self):
        np.random.shuffle(self.binned)

    def gen_dataset(self):
        colset = self.df.columns
        tmpdf = self.df[colset[:self.df.shape[1]-1]].dropna(axis=0)
        print("Smoothing dataset...")
        for i in range(2,150,3):
            tmpdf[['xint_rolling_'+str(i), 'yint_rolling_'+str(i)]] = self.df[['xint','yint']].rolling(i).mean()
        tmpdf = tmpdf.dropna(axis=0)
        tmpdf = (tmpdf-tmpdf.min()) /(tmpdf.max()-tmpdf.min())
        
        tmpdf['labels'] = self.df.loc[tmpdf.index, 'stim_onset']
        self.data = tmpdf.loc[:,['xint_rolling_14', 'xint_rolling_20', 'xint_rolling_29',
               'xint_rolling_35', 'xint_rolling_41', 'yint_rolling_41',
               'xint_rolling_116', 'yint_rolling_116', 'yint_rolling_119',
               'yint_rolling_122', 'xint_rolling_125', 'yint_rolling_125',
               'xint_rolling_128', 'yint_rolling_128', 'xint_rolling_131',
               'yint_rolling_131', 'xint_rolling_134', 'yint_rolling_134',
               'xint_rolling_137', 'yint_rolling_137', 'xint_rolling_140',
               'yint_rolling_140', 'xint_rolling_143', 'yint_rolling_143',
               'xint_rolling_146', 'yint_rolling_146', 'xint_rolling_149',
               'yint_rolling_149', 'labels']]
        self.datalist = self.data.columns[:-1]
        self.width = self.data.shape[1]-1
        self.end_point = tmpdf.shape[0]
        
        print('')

    def undersample(self):
        '''
        Takes in a binned matrix and the number of positive examples and balances
        the dataset thus that the number of negative examples are undersampled to 
        match the number of positive examples.
        '''
        tmpx = np.ones((self.n_pos*2, self.width+1, self.height))
        n_negative = 0
        j = 0
        for i in range(self.binned.shape[0]):
            if j == tmpx.shape[0]:
                # if we run out of tmpx slots, we stop
                continue

            if self.binned[i,-1,0] > 0:
                tmpx[j,:,:] = self.binned[i,:,:]
                j += 1
            elif self.binned[i,-1,0] == 0 & n_negative < self.n_pos:
                if random.uniform(0,1.0) < 0.35:
                    tmpx[j,:,:] = self.binned[i,:,:]
                    j += 1
                    n_negative += 0
            else:
                continue

        self.undersampled = tmpx


    def bin_data(self):
        tmpx = np.ones((self.end_point//self.height, self.width+1, self.height))
        self.n_pos = 0
        for i, t in enumerate(range(self.height,self.end_point+1,self.height)):
            tmpdf = self.data.iloc[t-self.height:t,:-1]
            tmpx[i,:self.width,:] = np.transpose(tmpdf.values)
            tmpx[i,self.width,:] = np.sum(self.data['labels'].loc[tmpdf.index])/self.height
            if tmpx[i,self.width,0] > 0.0:
                self.n_pos += 1
        np.random.shuffle(tmpx)
        # tmp1 = tmpx
        # np.random.shuffle(tmpx)
        # tmp2 = tmpx
        # np.random.shuffle(tmpx)
        # tmp3 = tmpx
        # tmpx = np.vstack((tmp1,tmp2,tmp3))
        self.binned = tmpx

    def gen_array(self):
        #self.choose_height()
        self.gen_dataset()
        self.bin_data()
            
    def get_processed_data(self):
        if len(self.binned) == 0:
            print("No binned arrays!")
        elif len(self.datalist) == 0:
            print("No column names!")
        elif type(self.undersampled) == type(None):
            return self.binned
        else:
            return self.undersampled

    def save_bins(self):
        for key in self.datalist:
            tmpdf = self.binned[key]
            exist = os.path.exists('./es_finder_res/h_'+str(self.height))
            if exist == False:
                os.mkdir('./es_finder_res/h_'+str(self.height))
            exist = os.path.exists('./es_finder_res/h_'+str(self.height)+'/binned')
            if exist==False:
                os.mkdir('./es_finder_res/h_'+str(self.height)+'/binned')
            tmpdf.to_csv('./es_finder_res/h_'+str(self.height)+'/binned/'+key+'.csv', dtype=np.float32)

class EBNet(nn.Module):
    def __init__(self):
        self.data = None
        self.seed = random.seed(52497)
        self.width = 2
        self.height = h
        self.kernel1 = k1
        self.kernel2 = k2
        self.kernel3 = k3
        self.flat_size = flat_size
        self.d1n = dn1
        self.d2n = dn2
        self.d3n = dn3
        self.state = {'kernel':[], 'dropout1':[], 'dropout2':[]}

        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, self.kernel1, padding='same', dtype=torch.double)
        # self.conv1s = nn.Conv2d(1, 5, self.kernel1, padding='same', dtype=torch.double)
        self.conv2 = nn.Conv2d(5, 15, self.kernel2, padding='same', dtype=torch.double)
        self.conv3 = nn.Conv2d(15, 37, self.kernel3, padding='same', dtype=torch.double)
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.d1 = nn.Dropout(self.d1n)
        self.d2 = nn.Dropout(self.d2n) #(0.37)
        self.d3 = nn.Dropout(self.d3n)

        self.fc1 = nn.Linear(self.flat_size, 500, dtype=torch.double)
        self.fc2 = nn.Linear(500, 250, dtype=torch.double)
        self.fc3 = nn.Linear(250, 100, dtype=torch.double)
        self.fc4 = nn.Linear(100, 10, dtype=torch.double)
        self.fc5 = nn.Linear(10, 1, dtype=torch.double)

    def choose_kernel(self):
        self.kernel1 = random.randint(5,self.height-10)
        self.kernel2 = random.randint(3, self.kernel1-1)
        self.kernel3 = random.randint(2, self.kernel2-1)
        self.state['kernel'].append([self.kernel1,self.kernel2,self.kernel3])

    def choose_dropout(self):
        self.d1n = random.uniform(.1,.7)
        self.d2n = random.uniform(.1,.7)
        self.state['dropout1'].append(self.d1n)
        self.state['dropout2'].append(self.d2n)

    def create_state(self):
        self.choose_kernel()
        self.choose_dropout()
        #self.__init__()

    def get_state(self):
        return self.get_state

    def forward(self, bin_arr):
        # conv_layers = []
        x = bin_arr
        # print("Forward convolutions: ")
        # print("...initial shape: ", x.shape)
        self.width = x.shape[2]
        self.height = x.shape[3]

        x = self.pool(F.relu(self.conv1(x)))
        x = self.d1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.d2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.d3(x)
        x = torch.flatten(x, 1) 

        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = F.relu(self.fc3(x))
        x = self.d3(x)
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc5(x), dim=1)
        return x

def prettified_xy(data, batch_size):
    bin_rx = []
    bin_ry = []
    bins = data.shape[0]
    sets = data.shape[1]
    print('Formatting data...')
    for i in range(batch_size, bins+1, batch_size):
        X = torch.from_numpy(data[i-batch_size:i, :-1, :]).unsqueeze(1).to(device=cuda, dtype=torch.double)
        y = torch.from_numpy(data[i-batch_size:i, -1, 0]).to(dtype=torch.double)
        y = y.to(device=cuda)

        bin_rx.append(X)
        bin_ry.append(y.unsqueeze(0))

    #print('shape should be batch x 106 x 30:', [x.shape for x in X])

    return bin_rx, bin_ry


# Running stuff
bdf = pd.read_csv("./bdf_9.csv", index_col=0, usecols=['e_time', 'xint', 'yint', 'xint_diff', 'yint_diff', 'xint_diff_2', 'yint_diff_2', 'stim_onset'], low_memory=True, dtype={'xint':np.float32, 'yint':np.float32, 'xint_diff':np.float32, 'yint_diff':np.float32, 'xint_diff_2':np.float32, 'yint_diff_2':np.float32, 'stim_onset':np.int16})

print("Creating training set...\n")


global cuda, k1, k2, k3, dn1, dn2, dn3
cuda = torch.device('cuda:0')
h = 25
k1 = 3
k2 = 5
k3 = 3
dn1 = 0.37
dn2 = 0.27
dn3 = 0.47
flat_size = 333

datamaster = DataCurator(bdf.iloc[:int(bdf.shape[0]*.70),:])
datamaster.set_height(h)
datamaster.gen_array()
print("...Undersampling negative examples...")

# datamaster.save_shuffled()

# For batching data
print("Sending to model...")
multinet = EBNet().to(cuda)

maxfinder = []
state_saver = []

print("...Setting loss function and optimizer...")
criterion = nn.MSELoss()
#
# for lr in [0.000001, 0.00001, 0.0001, 0.001, 0.01]:
#     for wd in [0.01, 0.001, 0.0001, 0.00001]:
lr=0.10
wd=0.01
optimizer = optim.AdamW(multinet.parameters(), lr=lr, weight_decay=wd)
# optimizer = optim.SGD(multinet.parameters(), lr=lr, momentum=0.7, weight_decay=wd)
print("......Running learning rate:%s, weight decay:%s, with height:%s...\n" % (str(lr),str(wd),str(h)))

maxi = 0

running_loss = 0.0
tmp = []

for k1 in np.arange(1,9,2):
    for k2 in np.arange(1,9,2):
        for k3 in np.arange(1,9,2):
            for dn1 in np.arange(0.001, 0.7, 0.001):
                for dn2 in np.arange(0.001, 0.7, 0.001):
                    for dn3 in np.arange(0.001, 0.7, 0.001):
                        for epoch in range(5):
                            batch_size=100
                            print("...Epoch: %s\n" %str(epoch))
                            print("......Shuffling data....")
                            print("......Model parameters for kernel 1: %s, kernel 2: %s, kernel 3: %s, dropout layer 1: %s, dropout layer 2: %s, and dropout layer 3: %s" 
    % (k1, k2, k3, dn1, dn2, dn3))
                            datamaster.shuffle_mat()
                            datamaster.undersample()
                            data = datamaster.get_processed_data()

                            rx, ry = prettified_xy(data, batch_size)
                            
                            for i in range(len(rx)):
                                # zero the parameter gradients
                                optimizer.zero_grad()

                                # print("Sending forward batch %s..." % str(i))
                                # forward + backward + optimize
                                # this is not a dictionary

                                outputs = multinet(rx[i]).to(device=cuda)

                                loss = criterion(outputs, ry[i])
                                # print("Sending backward batch %s...\n" % str(i))
                                loss.backward()
                                optimizer.step()

                                # print statistics
                                running_loss += loss.item()
                                print('...Batch: %s, loss: %s' % (i, loss.item()))
                                corratio = 1-(running_loss/((i+1)*(epoch+1)))
                                if corratio > maxi:
                                    maxi=corratio
                                    tmp.append(maxi)
                                    print("Current best accuracy: %s, batch number: %s" % (str(maxi),str(i)))
                        print("Overall best accuracy: %s" % str(max(tmp)))

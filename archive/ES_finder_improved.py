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
        self.data = None
        self.binned = {}
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

    def gen_dataset(self):
        colset = self.df.columns
        tmpdf = self.df[colset[:self.df.shape[1]-1]].dropna(axis=0)
        print("Smoothing dataset...")
        for i in range(2,150,3):
            tmpdf[['xint_rolling_'+str(i), 'yint_rolling_'+str(i)]] = self.df[['xint','yint']].rolling(i).mean()
        tmpdf = tmpdf.dropna(axis=0)
        tmpdf = (tmpdf-tmpdf.min()) /(tmpdf.max()-tmpdf.min())
        self.datalist = tmpdf.columns
        self.width = tmpdf.shape[1]
        tmpdf['labels'] = self.df.loc[tmpdf.index, 'stim_onset']
        self.data = tmpdf
        self.end_point = tmpdf.shape[0]
        
        print('')

    def bin_data(self):
        tmpx = np.ones((self.end_point//self.height, self.width+1, self.height))
        for i, t in enumerate(range(self.height,self.end_point+1,self.height)):
            tmpdf = self.data.iloc[t-self.height:t,:-1]
            tmpx[i,:self.width,:] = np.transpose(tmpdf.values)
            tmpx[i,self.width,:] = np.sum(self.data['labels'].loc[tmpdf.index])/self.height
        np.random.shuffle(tmpx)
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
        return self.binned

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
        self.height = 30
        self.kernel1 = 2
        self.kernel2 = 3
        self.kernel3 = 5
        self.flat_size = 51*40*3
        self.d1n = 0.37
        self.d2n = 0.57
        self.state = {'kernel':[], 'dropout1':[], 'dropout2':[]}

        super().__init__()
        self.conv1 = nn.Conv1d(6, 12, self.kernel1, padding=0, dtype=torch.double)
        self.conv1s = nn.Conv1d(2, 12, self.kernel1, padding=0, dtype=torch.double)
        self.conv2 = nn.Conv1d(12, 20, self.kernel2, padding=0, dtype=torch.double)
        self.conv3 = nn.Conv1d(20, 40, self.kernel3, padding=0, dtype=torch.double)
        
        self.pool = nn.MaxPool1d(2)
        
        self.d1 = nn.Dropout(self.d1n)
        self.d2 = nn.Dropout(self.d2n) #(0.37)
        
        self.fc1 = nn.Linear(self.flat_size, 500, dtype=torch.double)
        self.fc2 = nn.Linear(500, 100, dtype=torch.double)
        self.fc3 = nn.Linear(100, 10, dtype=torch.double)
        self.fc4 = nn.Linear(10, 1, dtype=torch.double)

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
        conv_layers = []

        for arr in bin_arr:
            x = arr
            # print("Forward convolutions: ")
            # print("...initial shape: ", x.shape)
            self.width = x.shape[1]
            self.height = x.shape[2]
            if self.height==0 or self.width < 2:
                print('--------*****SKIPPING*****----------')
                print(x)
                continue
            elif self.width==6:
                x = self.pool(F.relu(self.conv1(x)))
                # print("......shape after conv1 for 6 channels: ", x.shape)
            else:
                x = self.pool(F.relu(self.conv1s(x)))
                # print("......shape after conv1 for 6 channels: ", x.shape)
            x = self.pool(F.relu(self.conv2(x)))
            # print("......shape after conv2: ", x.shape)
            x = self.pool(F.relu(self.conv3(x)))
            # print("......shape after conv3: ", x.shape)
            x = torch.flatten(x, 1) 
            # print("......shape after flatten: ", x.shape)
            # flatten all dimensions except batch
            conv_layers.append(x)
        
        x = torch.cat(conv_layers, dim=1)
        # print("......shape after concatenation: ", x.shape)

        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x

def prettified_xy(data, batch_size):
    bin_rx = []
    bin_ry = []
    bins = data.shape[0]
    sets = data.shape[1]
    print('Formatting data...')
    for i in range(batch_size, bins+1, batch_size):
        X = [torch.from_numpy(data[i-batch_size:i, :6, :]).to(device=cuda, dtype=torch.double)]
        y = torch.from_numpy(data[i-batch_size:i, -1, 0]).to(dtype=torch.double)
        y = y.to(device=cuda)
            
        s = 6
        while s < sets-1:
            # print('In while-loop, creating the X tensor set for batch: %s, subset: %s of %s' % (str(i), str(s), str(sets)))
            xi = data[i-batch_size:i,s:s+2,:]
            X.append(torch.from_numpy(xi).to(device=cuda, dtype=torch.double))
            s += 2
        
        bin_rx.append(X)
        bin_ry.append(y.unsqueeze(0))

    #print('shape should be batch x 106 x 30:', [x.shape for x in X])

    return bin_rx, bin_ry

# Running stuff
bdf = pd.read_csv("./bdf_9.csv", index_col=0, usecols=['e_time', 'xint', 'yint', 'xint_diff', 'yint_diff', 'xint_diff_2', 'yint_diff_2', 'stim_onset'], low_memory=True, dtype={'xint':np.float32, 'yint':np.float32, 'xint_diff':np.float32, 'yint_diff':np.float32, 'xint_diff_2':np.float32, 'yint_diff_2':np.float32, 'stim_onset':np.int16})

print("Creating training set...\n")
# If shuffled form exists, load that; else create it
global cuda
cuda = torch.device('cuda:0')

h = 50
datamaster = DataCurator(bdf.iloc[:int(bdf.shape[0]*.70),:])
datamaster.set_height(h)
datamaster.gen_array()
data = datamaster.get_processed_data()
# datamaster.save_shuffled()

# For batching data
batch_size=100
rx, ry = prettified_xy(data, batch_size)
assert len(rx) == len(ry)

print("Sending to model...\n")
multinet = EBNet().to(cuda)

maxfinder = []
state_saver = []

print("Setting loss function and optimizer...\n")
criterion = nn.MSELoss()
#optimizer = optim.SGD(multinet.parameters(), lr=0.0001, momentum=0.7, weight_decay=0.0001)
for lr in [0.000001, 0.00001, 0.0001, 0.001, 0.01]:
    for wd in [0.01, 0.001, 0.0001, 0.00001]:
        optimizer = optim.AdamW(multinet.parameters(), lr=lr, weight_decay=wd)
        
        print("Running learning rate:%s, weight decay:%s, with height:%s...\n" % (str(lr),str(wd),str(h)))

        maxi = 0
        
        running_loss = 0.0
        tmp = []

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
            # print(loss.item())
            corratio = 1-(running_loss*1/(i+1))
            if corratio > maxi:
                maxi=corratio
                print("Current best accuracy: %s " % str(maxi))
    #     tmp.append(maxi)
    # maxfinder.append(tmp)

    #         print("Completed attempt %s! \n\n\n" % str(attempt))
    #         exit()
    #         datamaster.gen_array()
    #         data = datamaster.get_processed_data()

        # for attempt in range(1):

        #     print("Setting parameters for attempt %s, d1 = .37, d2 = .57...\n" % str(attempt))
        #     #multinet.create_state()
        #     # state_saver.append(multinet.get_state())

        #     for epoch in range(1): # loop over the dataset multiple times

        #         print("Running epoch %s...\n" % str(epoch))

        #         maxi = 0
                
        #         running_loss = 0.0
        #         tmp = []
        #         rx, ry = prettified_xy(data, batch_size)

        #         for i in range(len(rx)):
        #             # zero the parameter gradients
        #             optimizer.zero_grad()

        #             #print("Sending forward batch %s..." % str(i))
        #             # forward + backward + optimize
        #             # this is not a dictionary
        #             outputs = multinet(rx[i]).to(device=cuda)
        #             loss = criterion(outputs, ry[i])
        #             #print("Sending backward batch %s...\n" % str(i))
        #             loss.backward()
        #             optimizer.step()

        #             # print statistics
        #             running_loss += loss.item()
        #             corratio = 1-(running_loss*1/(i+1))
        #             if corratio > maxi:
        #                 maxi=corratio
        #                 print("Current best accuracy: %s " % str(maxi))
        #         tmp.append(maxi)
        #     maxfinder.append(tmp)

        #     print("Completed attempt %s! \n\n\n" % str(attempt))
        #     exit()
        #     datamaster.gen_array()
        #     data = datamaster.get_processed_data()
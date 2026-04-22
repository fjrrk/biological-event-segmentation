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
        self.data = {}
        self.binned = {}
        self.datalist = []
        self.shuffled = {}
        self.past_heights = []

    def get_past_heights(self):
        return self.past_heights

    def get_data_dict(self):
        return self.data

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
        tmpdf = (tmpdf-tmpdf.min()) /(tmpdf.max()-tmpdf.min())
        tmpdf['labels'] = self.df.loc[tmpdf.index, 'stim_onset']
        self.data['x_f'] = tmpdf#.values
        self.datalist.append('x_f')
        steps = []
        # steps = [30]
        # while steps[-1] + steps[-1]*2 < 1000:
        #     steps.append(steps[-1]*2)
        print("Smoothing dataset...")
        for i in range(2,150,3):
            steps.append(i)
            tmpdf = self.df[['xint','yint']].rolling(i).mean()

            self.df[['xint_rolling_'+str(i), 'yint_rolling_'+str(i)]] = np.empty((self.df.shape[0], 2))*np.nan
            self.df[['xint_rolling_'+str(i), 'yint_rolling_'+str(i)]] = self.df[['xint','yint']].rolling(i).mean()
            
            tmpdf = self.df[['xint_rolling_'+str(i), 'yint_rolling_'+str(i)]].dropna(axis=0)
            # print("Smoothing factor: %s, dataframe height: %s" % (str(i), str(tmpdf.shape[0])))
            if tmpdf.shape[0]>20:
                tmpdf = (tmpdf-tmpdf.min()) /(tmpdf.max()-tmpdf.min())
                tmpdf['labels'] = self.df.loc[tmpdf.index, 'stim_onset']
                self.data['x_'+str(i)] = tmpdf#.values
                self.datalist.append('x_'+str(i))
        print('')

    def bin_data(self):
        for key in self.datalist:
            height = self.height
            # Note that self.data is a dict and key is the key, self.data[key] is a df
            width = self.data[key].shape[1]-1
            end_point = self.data[key].shape[0]

            tmpx = np.ones((end_point//height, width+1, height))

            for i, t in enumerate(range(height,end_point+1,height)):
                tmpx[i,:width,:] = np.transpose(self.data[key].iloc[t-height:t,:width].values)
                tmpx[i,width,:] = np.sum(self.data[key].iloc[t-height:t,width])/height
            np.random.shuffle(tmpx)
            self.binned[key] = tmpx

    def gen_array(self):
        #self.choose_height()
        self.gen_dataset()
        self.bin_data()
    
    def set_shuffled_tensors(self):
        if len(self.binned) == 0:
            print("No binned arrays!")
        elif len(self.datalist) == 0:
            print("No column names!")

        for key in self.datalist:
            tmpx = self.binned[key]
            np.random.shuffle(tmpx)
            tmpx = torch.from_numpy(tmpx)
            tmpx = tmpx.to(dtype=torch.double)
            self.shuffled[key] = tmpx            

    def get_processed_data(self):
        if len(self.binned) == 0:
            print("No binned arrays!")
        elif len(self.datalist) == 0:
            print("No column names!")
        self.set_shuffled_tensors()
        return self.shuffled

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

    def save_shuffled(self):
        for key in self.datalist:
            tmpdf = self.shuffled[key]
            exist = os.path.exists('./es_finder_res/h_'+str(self.height))
            if exist == False:
                os.mkdir('./es_finder_res/h_'+str(self.height))

            exist = os.path.exists('./es_finder_res/h_'+str(self.height)+'/shuffled')
            if exist==False:
                os.mkdir('./es_finder_res/h_'+str(self.height)+'/shuffled')
            torch.save(tmpdf, './es_finder_res/h_'+str(self.height)+'/shuffled/'+key+'.pt')

class EBNet(nn.Module):
    def __init__(self):
        self.data = None
        self.seed = random.seed(52497)
        self.width = 6
        self.height = 30
        self.kernel1 = 7
        self.kernel2 = 5
        self.kernel3 = 3
        self.flat_size = 120
        self.d1n = 0.37
        self.d2n = 0.57
        self.state = {'kernel':[], 'dropout1':[], 'dropout2':[]}

        super().__init__()
        self.conv1 = nn.Conv1d(6, 12, self.kernel1, padding='same', dtype=torch.double)
        self.conv1s = nn.Conv1d(2, 12, self.kernel1, padding='same', dtype=torch.double)
        self.conv2 = nn.Conv1d(12, 20, self.kernel2, padding='same', dtype=torch.double)
        self.conv3 = nn.Conv1d(20, 40, self.kernel3, padding='same', dtype=torch.double)
        
        self.pool = nn.MaxPool1d(2)
        
        self.d1 = nn.Dropout(self.d1n)
        self.d2 = nn.Dropout(self.d2n) #(0.37)
        
        self.fc1 = nn.Linear(6120, 500, dtype=torch.double)
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
            self.width = x.shape[1]
            self.height = x.shape[2]
            if self.height==0 or self.width==0:
                print('--------*****SKIPPING*****----------')
                continue
            elif self.width==6:
                x = self.pool(F.relu(self.conv1(x)))
            else:
                x = self.pool(F.relu(self.conv1s(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            print('Post convolution shape: %s:' % str(x.shape))
            x = torch.flatten(x, 1) 
            print('Post flattening shape: %s' % str(x.shape))
            if x.shape[0]==0 | x.shape[1]==0:
                print(x)
                continue
            # flatten all dimensions except batch
            conv_layers.append(x)
        
        x = torch.cat(conv_layers, dim=1)
        self.flat_size = x.shape[1]

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
    height = data['x_f'].shape[0]
    for i in range(batch_size, height-batch_size+1, batch_size):
        X = []
        y = []
        for key in data.keys():
            size = data[key].shape[1]
            # get the inputs
            Xi = data[key][i-batch_size:i,:size-1,:]
            Xi = Xi.to(dtype=torch.double)
            Xi = Xi.to(device=cuda)
            X.append(Xi)

        y = data[key][i-batch_size:i,size-1,:][:,0]
        y = y.to(dtype=torch.double)
        y = y.to(device=cuda)
            

        bin_rx.append(X)
        bin_ry.append(y.unsqueeze(0))
    return bin_rx, bin_ry

# Running stuff
bdf = pd.read_csv("./bdf_9.csv", index_col=0, usecols=['e_time', 'xint', 'yint', 'xint_diff', 'yint_diff', 'xint_diff_2', 'yint_diff_2', 'stim_onset'], low_memory=True, dtype={'xint':np.float32, 'yint':np.float32, 'xint_diff':np.float32, 'yint_diff':np.float32, 'xint_diff_2':np.float32, 'yint_diff_2':np.float32, 'stim_onset':np.int16})

print("Creating training set...\n")
# If shuffled form exists, load that; else creat it

for h in range(30, 200, 5):
    datamaster = DataCurator(bdf.iloc[:int(bdf.shape[0]*.70),:])
    datamaster.set_height(h)
    datamaster.gen_array()
    data = datamaster.get_processed_data()
    # datamaster.save_shuffled()

    # data = pd.read_csv('./bdf_shuffled_')
    print("Sending to model...\n")
    cuda = torch.device('cuda:0')
    multinet = EBNet().to(cuda)

    batch_size=1
    maxfinder = []
    state_saver = []

    print("Setting loss function and optimizer...\n")
    criterion = nn.MSELoss()
    #optimizer = optim.SGD(multinet.parameters(), lr=0.0001, momentum=0.7, weight_decay=0.0001)
    for lr in [0.000001, 0.00001, 0.0001, 0.001, 0.01]:
        for wd in [0.01, 0.001, 0.0001, 0.00001]:
            optimizer = optim.AdamW(multinet.parameters(), lr=0.0001, weight_decay=0.0001)
            
            print("Running learing rate:%s, weight decay:%s, with height:%s...\n" % (str(lr),str(wd),str(h)))

            maxi = 0
            
            running_loss = 0.0
            tmp = []
            rx, ry = prettified_xy(data, batch_size)

            for i in range(len(rx)):
                # zero the parameter gradients
                optimizer.zero_grad()

                #print("Sending forward batch %s..." % str(i))
                # forward + backward + optimize
                # this is not a dictionary
                outputs = multinet(rx[i]).to(device=cuda)
                if outputs.shape[0]<1:
                    print('Output shape has no rows')
                    continue
                loss = criterion(outputs, ry[i])
                #print("Sending backward batch %s...\n" % str(i))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
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

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
import math 
import random

from HelperClass2.NeuralNet_2_0 import *

train_data_name = "../../Data/ch09.train.npz"
test_data_name = "../../Data/ch09.test.npz"

def train(hp, folder):
    net = NeuralNet_2_0(hp, folder)
    net.train(dataReader, 50, True)
    trace = net.GetTrainingHistory()
    return trace


def ShowLossHistory(folder, file1, hp1, file2, hp2, file3, hp3, file4, hp4):
    lh = TrainingHistory_2_0.Load(file1)
    axes = plt.subplot(2,2,1)
    lh.ShowLossHistory4(axes, hp1)
    
    lh = TrainingHistory_2_0.Load(file2)
    axes = plt.subplot(2,2,2)
    lh.ShowLossHistory4(axes, hp2)

    lh = TrainingHistory_2_0.Load(file3)
    axes = plt.subplot(2,2,3)
    lh.ShowLossHistory4(axes, hp3)

    lh = TrainingHistory_2_0.Load(file4)
    axes = plt.subplot(2,2,4)
    lh.ShowLossHistory4(axes, hp4)

    plt.show()


def try_hyperParameters(folder, n_hidden, batch_size, eta):
    hp = HyperParameters_2_0(1, n_hidden, 1, eta, 10000, batch_size, 0.001, NetType.Fitting, InitialMethod.Xavier)
    filename = str.format("{0}\\{1}_{2}_{3}.pkl", folder, n_hidden, batch_size, eta).replace('.', '', 1)
    file = Path(filename)
    if file.exists():
        return file, hp
    else:
        lh = train(hp, folder)
        lh.Dump(file)
        return file, hp

def log_number_of_hidden_units(min_num,max_num,sample_num,distribution='uniform'):
    log_ne_list = np.random.uniform(low=math.log10(min_num),high=math.log10(max_num),size = sample_num)
    ne_list = np.power(10,log_ne_list)
    ne_list =  ne_list.astype(np.int64)
    #print("log_ne_list:",log_ne_list)
    #print("ne_list:",ne_list)
    return ne_list

def log_number_of_learn_rate(min_num,max_num,sample_num,distribution='uniform'):
    log_list = np.random.uniform(low=int(math.log10(min_num)),high=int(math.log10(max_num)),size = sample_num)
    lr_list = np.power(10,log_list)
    #print("log_list",log_list)
    #print("lr_list",lr_list)
    return lr_list
def exp_number_of_batch_size(min_num,max_num,sample_num,distribution='uniform'):
    exp_list = np.random.randint(low=math.log2(min_num),high=math.log2(max_num),size = sample_num)
    bs_list = np.power(2,exp_list)
    #print("exp_list",exp_list)
    #print("bs_list",bs_list)
    return bs_list

if __name__ == '__main__':
  
    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.GenerateValidationSet()
    
    folder = "complex_turn"

    sample_num = 4
    neParamSet = log_number_of_hidden_units(10,100,sample_num)
    batchParamSet = exp_number_of_batch_size(2,128,sample_num)
    etaParamSet = log_number_of_learn_rate(0.1,1,sample_num)
    print("neParamSet:",neParamSet)
    print("batchParamSet",batchParamSet)
    print("etaParamSet",etaParamSet)
    file_list = []
    hp_list = []
    for i in range(0,sample_num):
        file_temp, hp_temp = try_hyperParameters(folder, neParamSet[i], batchParamSet[i], etaParamSet[i])
        file_list.append(file_temp)
        hp_list.append(hp_temp)
    ShowLossHistory(folder, file_list[0], hp_list[0], file_list[1], hp_list[1], file_list[2], hp_list[2], file_list[3], hp_list[3])
   


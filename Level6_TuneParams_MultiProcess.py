
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
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


if __name__ == '__main__':
  
    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.GenerateValidationSet()
    
    folder = "complex_turn"

    start = time.time()
   
    result = []
    total_group = 12
    p = Pool(total_group)

    result.append(p.apply_async(try_hyperParameters, args=(folder, 4, 10, 0.1,)))
    result.append(p.apply_async(try_hyperParameters, args=(folder, 4, 10, 0.3,)))
    result.append(p.apply_async(try_hyperParameters, args=(folder, 4, 10, 0.5,)))
    result.append(p.apply_async(try_hyperParameters, args=(folder, 4, 10, 0.7,)))

    result.append(p.apply_async(try_hyperParameters, args=(folder, 4, 5, 0.5,)))
    result.append(p.apply_async(try_hyperParameters, args=(folder, 4, 10, 0.5,)))
    result.append(p.apply_async(try_hyperParameters, args=(folder, 4, 15, 0.5,)))
    result.append(p.apply_async(try_hyperParameters, args=(folder, 4, 20, 0.5,)))

    result.append(p.apply_async(try_hyperParameters, args=(folder, 2, 10, 0.5,)))
    result.append(p.apply_async(try_hyperParameters, args=(folder, 4, 10, 0.5,)))
    result.append(p.apply_async(try_hyperParameters, args=(folder, 6, 10, 0.5,)))
    result.append(p.apply_async(try_hyperParameters, args=(folder, 8, 10, 0.5,)))

    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    for i in range(0,int(total_group/4)):
        ShowLossHistory(folder, result[0+i*4].get()[0], result[0+i*4].get()[1],result[1+i*4].get()[0], result[1+i*4].get()[1], result[2+i*4].get()[0], result[2+i*4].get()[1], result[3+i*4].get()[0], result[3+i*4].get()[1])

    end = time.time()
    print("cost time %f s"%(end-start))




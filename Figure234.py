#!/usr/bin/env python
# coding: utf-8


import keras
import numpy as np
import time
import warnings

from eigenpro import kernels
from eigenpro import mnist
from eigenpro import ciphar
from eigenpro import synthetic
from eigenpro import utils
from eigenpro import training


### Dataset

dataset_dict = {}

#num_classes = 10
#(x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load()
#y_train_full = keras.utils.to_categorical(y_train_full, num_classes)
#y_test_full = keras.utils.to_categorical(y_test_full, num_classes)
#dataset = ((x_train_full, y_train_full), (x_test_full, y_test_full))
#dataset_dict['MNIST'] = dataset

#num_classes = 10
#(x_train_full, y_train_full), (x_test_full, y_test_full) = ciphar.load()
#y_train_full = keras.utils.to_categorical(y_train_full, num_classes)
#y_test_full = keras.utils.to_categorical(y_test_full, num_classes)
#dataset = ((x_train_full, y_train_full), (x_test_full, y_test_full))
#dataset_dict['CIPHAR'] = dataset

#num_classes = 2
#(x_train_full, y_train_full), (x_test_full, y_test_full) = synthetic.load(1)
#y_train_full = keras.utils.to_categorical(y_train_full, num_classes)
#y_test_full = keras.utils.to_categorical(y_test_full, num_classes)
#dataset = ((x_train_full, y_train_full), (x_test_full, y_test_full))
#dataset_dict['Synthetic1'] = dataset

num_classes = 2
(x_train_full, y_train_full), (x_test_full, y_test_full) = synthetic.load(2)
y_train_full = keras.utils.to_categorical(y_train_full, num_classes)
y_test_full = keras.utils.to_categorical(y_test_full, num_classes)
dataset = ((x_train_full, y_train_full), (x_test_full, y_test_full))
dataset_dict['Synthetic2'] = dataset



### Kernel

kernel_dict = {}

s = 5
kernel_sgd = lambda x,y: kernels.Gaussian(x, y, s)
kernel_inv = lambda x,y: training.Gaussian(x, y, s)
kernel_dict["Gaussian"] = (kernel_sgd, kernel_inv)


### Size 

# size_list = [200, 400]
#size_list = [2000, 4000, 10000, 20000, 50000]
size_list = [400, 1000, 2000, 4000, 10000, 15000, 20000, 25000, 30000, 50000]

### Noise

noise_list = [0, 10]


### Training

trainers = training.training(dataset_dict, kernel_dict, size_list, noise_list)

with open('output/figure234-' + time.strftime("%Y%m%d-%H%M%S") + '.txt', 'w') as f:
    print(trainers, file=f)


print()
print()
print()
print(trainers)

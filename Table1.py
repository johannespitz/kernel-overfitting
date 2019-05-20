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

num_classes = 10
(x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load()
y_train_full = keras.utils.to_categorical(y_train_full, num_classes)
y_test_full = keras.utils.to_categorical(y_test_full, num_classes)
dataset = ((x_train_full, y_train_full), (x_test_full, y_test_full))
dataset_dict['MNIST'] = dataset

num_classes = 10
(x_train_full, y_train_full), (x_test_full, y_test_full) = ciphar.load()
y_train_full = keras.utils.to_categorical(y_train_full, num_classes)
y_test_full = keras.utils.to_categorical(y_test_full, num_classes)
dataset = ((x_train_full, y_train_full), (x_test_full, y_test_full))
dataset_dict['CIPHAR'] = dataset

num_classes = 2
(x_train_full, y_train_full), (x_test_full, y_test_full) = synthetic.load(1)
y_train_full = keras.utils.to_categorical(y_train_full, num_classes)
y_test_full = keras.utils.to_categorical(y_test_full, num_classes)
dataset = ((x_train_full, y_train_full), (x_test_full, y_test_full))
dataset_dict['Synthetic1'] = dataset

num_classes = 2
(x_train_full, y_train_full), (x_test_full, y_test_full) = synthetic.load(2)
y_train_full = keras.utils.to_categorical(y_train_full, num_classes)
y_test_full = keras.utils.to_categorical(y_test_full, num_classes)
dataset = ((x_train_full, y_train_full), (x_test_full, y_test_full))
dataset_dict['Synthetic2'] = dataset



### Kernel

kernel_dict = {}

sg = 5
kernel_sgd = lambda x,y: kernels.Gaussian(x, y, sg)
kernel_inv = lambda x,y: training.Gaussian(x, y, sg)
kernel_dict["Gaussian"] = (kernel_sgd, kernel_inv)

sl = 10
kernel_sgd_l = lambda x,y: kernels.Laplace(x, y, sl)
kernel_inv_l = lambda x,y: training.Laplace(x, y, sl)
kernel_dict["Laplace"] = (kernel_sgd_l, kernel_inv_l)



### Size 

# size_list = [400]
size_list = [30000] # or [60000]



### Noise

noise_list = [0, 100]


### Training

trainers = training.training(dataset_dict, kernel_dict, size_list, noise_list, MAXEPOCH=300)

with open('output/table1-' + time.strftime("%Y%m%d-%H%M%S") + '.txt', 'w') as f:
    print(trainers, file=f)
    

print()
print()
print()
print(trainers)

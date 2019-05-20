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



### Kernel

kernel_dict = {}

sg = np.float(3.6)
kernel_sgd_g1 = lambda x,y: kernels.Gaussian(x, y, sg)
kernel_inv_g1 = lambda x,y: training.Gaussian(x, y, sg)
#kernel_dict["Gaussian *1"] = (kernel_sgd_g1, kernel_inv_g1)

kernel_sgd_g05 = lambda x,y: kernels.Gaussian(x, y, sg)
kernel_inv_g05 = lambda x,y: training.Gaussian(x, y, sg)
kernel_dict["Gaussian /2"] = (kernel_sgd_g05, kernel_inv_g05)

kernel_sgd_g2 = lambda x,y: kernels.Gaussian(x, y, sg * 2)
kernel_inv_g2 = lambda x,y: training.Gaussian(x, y, sg * 2)
#kernel_dict["Gaussian *2"] = (kernel_sgd_g2, kernel_inv_g2)


sl = 10
kernel_sgd_l1 = lambda x,y: kernels.Laplace(x, y, sl)
kernel_inv_l1 = lambda x,y: training.Laplace(x, y, sl)
#kernel_dict["Laplace *1"] = (kernel_sgd_l1, kernel_inv_l1)

kernel_sgd_l05 = lambda x,y: kernels.Laplace(x, y, sl / 2)
kernel_inv_l05 = lambda x,y: training.Laplace(x, y, sl / 2)
#kernel_dict["Laplace /2"] = (kernel_sgd_l05, kernel_inv_l05)

kernel_sgd_l2 = lambda x,y: kernels.Laplace(x, y, sl * 2)
kernel_inv_l2 = lambda x,y: training.Laplace(x, y, sl * 2)
#kernel_dict["Laplace *2"] = (kernel_sgd_l2, kernel_inv_l2)



### Size 

# size_list = [200]
size_list = [10000] # or [60000]



### Noise

noise_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# noise_list = [0, 33, 67, 100]


### Training

trainers = training.training(dataset_dict, kernel_dict, size_list, noise_list)

with open('output/figure8-' + time.strftime("%Y%m%d-%H%M%S") + '.txt', 'w') as f:
    print(trainers, file=f)
    

print()
print()
print()
print(trainers)

#!/usr/bin/env python
# coding: utf-8


# In[28]:


'''
    Modified version of https://github.com/EigenPro/EigenPro-tensorflow
    in particular run_expr.py
'''

from __future__ import print_function

import keras
import numpy as np
import time
import warnings

from dataclasses import dataclass
from scipy.stats import bernoulli

from distutils.version import StrictVersion
from keras.layers import Dense, Input
from keras.models import Model
from keras import backend as K

from eigenpro import kernels
from eigenpro import mnist
from eigenpro import ciphar
from eigenpro import synthetic
from eigenpro import utils

from eigenpro.backend_extra import hasGPU
from eigenpro.layers import KernelEmbedding, RFF
from eigenpro.optimizers import PSGD, SGD

assert StrictVersion(keras.__version__) >= StrictVersion('2.0.8'),        "Requires Keras (>=2.0.8)."

if StrictVersion(keras.__version__) > StrictVersion('2.0.8'):
    warnings.warn('\n\nEigenPro-tensorflow has been tested with Keras 2.0.8. '
                   'If the\ncurrent version (%s) fails, ' 
                   'switch to 2.0.8 by command,\n\n'
                   '\tpip install Keras==2.0.8\n\n' %(keras.__version__), Warning)

assert keras.backend.backend() == u'tensorflow',        "Requires Tensorflow (>=1.2.1)."
assert hasGPU(), "Requires GPU."


# In[29]:


'''
    Modified version of https://github.com/EigenPro/EigenPro-tensorflow
    in particular kernels.py
'''

def D2(X, Y):
    XX = np.sum(np.square(X), axis = 1, keepdims=True)
    if X is Y:
        YY = XX
    else:
        YY = np.sum(np.square(Y), axis = 1, keepdims=True)
    XY = np.dot(X, np.transpose(Y))
    d2 = np.reshape(XX, (np.shape(X)[0], 1))        + np.reshape(YY, (1, np.shape(Y)[0]))        - 2 * XY
    return d2

def Gaussian(X, Y, s):
    assert s > 0    
    d2 = D2(X, Y)
    gamma = np.float32(1. / (2 * s ** 2))
    G = np.exp(-gamma * np.clip(d2, 0, None))
    return G

def Laplace(X, Y, s):
    assert s > 0
    d2 = np.clip(D2(X, Y), 0, None)
    d = np.sqrt(d2)
    G = np.exp(- d / s)
    return G


# In[30]:


def add_noise(y, noise):
    n, dim = y.shape
    y = np.argmax(y, axis=1)
    change = np.array(bernoulli.rvs(noise / 100, size=n))
    change = change * np.random.randint(dim, size=n)
    y = np.mod(y + change, dim)
    if noise == 100:
        y = np.random.randint(dim, size=n)
    return keras.utils.to_categorical(y, dim)

def my_norm(alpha, K):   
    cross_norm = alpha.T.dot(K).dot(alpha)
    return np.sum(np.sqrt(np.diag(cross_norm)))   


# In[38]:


def training(data_set_dict, kernel_dict, size_list, noise_list, MAXEPOCH=100):
    
    trainers = {}    
    
    for dataset_name, ((x_train_full, y_train_full), (x_test_full, y_test_full)) in data_set_dict.items():
        _, num_classes = y_train_full.shape
        
        for kernel_name, (kernel_sgd, kernel_inv) in kernel_dict.items():

            for size in size_list:

                for noise in noise_list:

                    name = 'D:' + dataset_name + ' K:' + kernel_name + ' S:' + str(size) + ' N:' + str(noise)
                    print(name)
                    trainer = {'dataset': dataset_name, 'kernel': kernel_name, 'size': size, 'noise': noise}

                    x_train = x_train_full[0:size]
                    x_test  = x_test_full

                    y_train = add_noise(y_train_full[0:size], noise)        
                    y_test  = add_noise(y_test_full, noise)

                    # Set the hyper-parameters.
                    bs = 256                # size of the mini-batch
                    M = min(size, 5000)     # (EigenPro) subsample size
                    k = min(size - 1, 160)  # (EigenPro) top-k eigensystem

                    n, D = x_train.shape    # (n_sample, n_feature)

                    # Calculate step size and (Primal) EigenPro preconditioner.
                    kf, scale, s0 = utils.asm_eigenpro_f(
                        x_train, kernel_sgd, M, k, 1, in_rkhs=True)
                    eta = np.float32(1.5 / s0) # 1.5 / s0
                    eta = eta * num_classes # correction due to mse loss

                    input_shape = (D+1,) # n_feature, (sample) index
                    ix = Input(shape=input_shape, dtype='float32', name='indexed-feat-')
                    x, index = utils.separate_index(ix) # features, sample_id
                    kfeat = KernelEmbedding(kernel_sgd, x_train, input_shape=(D,))(x)

                    # Assemble kernel EigenPro trainer.
                    y = Dense(num_classes, input_shape=(n,),
                              activation='linear',
                              kernel_initializer='zeros',
                              use_bias=False,
                              name='trainable')(kfeat)        
                    model = Model(ix, y)
                    model.compile(loss='mse', optimizer=PSGD(pred_t=y,
                                                 index_t=index,
                                                 eta=scale*eta,
                                                 eigenpro_f=lambda g: kf(g, kfeat)),
                                                 metrics=['accuracy'])        

                    model.summary(print_fn=print)
                    print()

                    initial_epoch=0
                    np.random.seed(1) # Keras uses numpy random number generator
                    train_ts = 0 # training time in seconds

                    print("Stochastic Gradient Descent")
                    for epoch in range(1, MAXEPOCH + 1):

                        start = time.time()
                        model.fit(
                            utils.add_index(x_train), y_train,
                            batch_size=bs, epochs=epoch, verbose=0,
                            validation_data=(utils.add_index(x_test), y_test),
                            initial_epoch=initial_epoch)
                        train_ts += time.time() - start
                        tr_score = model.evaluate(utils.add_index(x_train), y_train, verbose=0)
                        te_score = model.evaluate(utils.add_index(x_test), y_test, verbose=0)
                        initial_epoch = epoch

                        if tr_score[1] == 1.0:                           
                            trainer['sgd_ce'] = 1 - te_score[1]
                            trainer['iterations'] = epoch
                            print("train error: %.2f%%\ttest error: %.2f%% (%d epochs, %.2f seconds)" %
                                 ((1 - tr_score[1]) * 100, (1 - te_score[1]) * 100, epoch, train_ts))
                            print("Zero Train Error")
                            print()    
                            break

                        if epoch == MAXEPOCH:                
                            trainer['sgd_ce'] = 1 - te_score[1]
                            trainer['iterations'] = 999999
                            print("train error: %.2f%%\ttest error: %.2f%% (%d epochs, %.2f seconds)" %
                                 ((1 - tr_score[1]) * 100, (1 - te_score[1]) * 100, epoch, train_ts))
                            print("Did not reach Zero Train Error")
                            print()
                            break

                        if epoch % 5 == 1:
                            print("train error: %.2f%%\ttest error: %.2f%% (%d epochs, %.2f seconds)" %
                                 ((1 - tr_score[1]) * 100, (1 - te_score[1]) * 100, epoch, train_ts))

                    alpah_sgd = np.array(model.get_layer("trainable").get_weights()[0])
                    
                    del model
                    utils.reset()

                    # linear system        

                    K_train = kernel_inv(x_train, x_train)
                    
                    if size <= 20000:
                        alpha = np.linalg.solve(K_train, y_train)
                        ## this was a test -> alpha and the trainable layer are interchangable
                        # alpha = model.get_layer("trainable").get_weights()[0]
                        K_test = kernel_inv(x_train, x_test)
                        pred = K_test.T.dot(alpha)
                        miss_count = np.count_nonzero(np.argmax(pred, axis=1) - np.argmax(y_test, axis=1)) 
                        miss_rate = miss_count / y_test.shape[0]

                        trainer['inv_ce'] = miss_rate

                        print("Linear Interpolation")
                        print("Classification Error = " + str(miss_rate))
                        print()  

                        trainer['inv_norm'] = my_norm(alpha, K_train)
                        
                    trainer['sgd_norm'] = my_norm(alpah_sgd, K_train)
                    trainers[name] = trainer  

                    K_train = None
                    K_test = None
                    utils.reset()

    print("Done")
    return trainers




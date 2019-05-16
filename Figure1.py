'''
    Modified version of https://github.com/EigenPro/EigenPro-tensorflow
    in particular run_expr.py
'''

from __future__ import print_function

import argparse
import collections
import keras
import numpy as np
import time
import warnings

from distutils.version import StrictVersion
from keras.layers import Dense, Input
from keras.models import Model
from keras import backend as K

from eigenpro import kernels
from eigenpro import mnist
from eigenpro import ciphar
from eigenpro import synthetic
from eigenpro import utils
from eigenpro import training

from eigenpro.backend_extra import hasGPU
from eigenpro.layers import KernelEmbedding, RFF
from eigenpro.optimizers import PSGD, SGD

assert StrictVersion(keras.__version__) >= StrictVersion('2.0.8'), \
       "Requires Keras (>=2.0.8)."

if StrictVersion(keras.__version__) > StrictVersion('2.0.8'):
    warnings.warn('\n\nEigenPro-tensorflow has been tested with Keras 2.0.8. '
                   'If the\ncurrent version (%s) fails, ' 
                   'switch to 2.0.8 by command,\n\n'
                   '\tpip install Keras==2.0.8\n\n' %(keras.__version__), Warning)

assert keras.backend.backend() == u'tensorflow', \
       "Requires Tensorflow (>=1.2.1)."
assert hasGPU(), "Requires GPU."


# Set the hyper-parameters.
bs = 256            # size of the mini-batch
M = 5000            # (EigenPro) subsample size
k = 160             # (EigenPro) top-k eigensystem


for dataset in ['MNIST', 'CIPHAR', 'Synthetic1', 'Synthetic2']:
# for dataset in ['MNIST']:

   if dataset is 'MNIST':
      num_classes = 10  
      (x_train, y_train), (x_test, y_test) = mnist.load()
   elif dataset is 'CIPHAR':
      num_classes = 10  
      (x_train, y_train), (x_test, y_test) = ciphar.load()
   elif dataset is 'Synthetic1':
      num_classes = 2
      (x_train, y_train), (x_test, y_test) = synthetic.load(1)
   elif dataset is 'Synthetic2':
      num_classes = 2
      (x_train, y_train), (x_test, y_test) = synthetic.load(2)

   # ### NOGPU
   # size = 1000
   # x_train = x_train[0:size]
   # y_train = y_train[0:size]
   # M = 400
   # ### NOGPU End

   n, D = x_train.shape    # (n_sample, n_feature)

   # convert class vectors to binary class matrices
   y_train = keras.utils.to_categorical(y_train, num_classes)
   y_test = keras.utils.to_categorical(y_test, num_classes)


   trainers = collections.OrderedDict()
   Trainer = collections.namedtuple('Trainer', ['model', 'x_train', 'x_test', 'tr_scores', 'te_scores'])

   input_shape = (D+1,) # n_feature, (sample) index
   ix = Input(shape=input_shape, dtype='float32', name='indexed-feat')
   x, index = utils.separate_index(ix)	# features, sample_id

   ## Gauss

   # Calculate step size and (Primal) EigenPro preconditioner.
   s = 5   # kernel bandwidth
   kernel = lambda x,y: kernels.Gaussian(x, y, s)
   kfeat = KernelEmbedding(kernel, x_train, input_shape=(D,))(x)
   kf, scale, s0 = utils.asm_eigenpro_f(
      x_train, kernel, M, k, 1, in_rkhs=True)
   eta = np.float32(1.5 / s0) # 1.5 / s0
   eta = eta * num_classes # correction due to mse loss

   # Assemble kernel EigenPro trainer.
   y = Dense(num_classes, input_shape=(n,),
            activation='linear',
            kernel_initializer='zeros',
            use_bias=False)(kfeat)
   model = Model(ix, y)
   model.compile(loss='mse',
               optimizer=PSGD(pred_t=y,
                              index_t=index,
                              eta=scale*eta,
                              eigenpro_f=lambda g: kf(g, kfeat)),
               metrics=['accuracy'])
   trainers['Gauss'] = Trainer(model=model,
                                 x_train = utils.add_index(x_train),
                                 x_test=utils.add_index(x_test),
                                 tr_scores={},
                                 te_scores={})


   ## Laplace

   # Calculate step size and (Primal) EigenPro preconditioner.
   s = np.float32(10)
   kernel = lambda x,y: kernels.Laplace(x, y, s)
   kfeat = KernelEmbedding(kernel, x_train, input_shape=(D,))(x)
   kf, scale, s0 = utils.asm_eigenpro_f(
      x_train, kernel, M, k, 1, in_rkhs=True)
   eta = np.float32(1.5 / s0) # 1.5 / s0
   eta = eta * num_classes # correction due to mse loss

   # Assemble kernel EigenPro trainer.
   y = Dense(num_classes, input_shape=(n,),
            activation='linear',
            kernel_initializer='zeros',
            use_bias=False)(kfeat)
   model = Model(ix, y)
   model.compile(loss='mse',
               optimizer=PSGD(pred_t=y,
                              index_t=index,
                              eta=scale*eta,
                              eigenpro_f=lambda g: kf(g, kfeat)),
               metrics=['accuracy'])
   trainers['Laplace'] = Trainer(model=model,
                                 x_train = utils.add_index(x_train),
                                 x_test=utils.add_index(x_test),
                                 tr_scores={},
                                 te_scores={})



   # Start training.
   for name, trainer in trainers.items():   
      print("")
      initial_epoch=0
      np.random.seed(1) # Keras uses numpy random number generator
      train_ts = 0 # training time in seconds
      for epoch in [1, 2, 5, 10, 20]:
         start = time.time()
         trainer.model.fit(
               trainer.x_train, y_train,
               batch_size=bs, epochs=epoch, verbose=0,
               validation_data=(trainer.x_test, y_test),
               initial_epoch=initial_epoch)
         train_ts += time.time() - start
         tr_score = trainer.model.evaluate(trainer.x_train, y_train, verbose=0)
         te_score = trainer.model.evaluate(trainer.x_test, y_test, verbose=0)
         trainer.tr_scores[epoch] = tr_score
         trainer.te_scores[epoch] = te_score
         print("%s\t\ttrain error: %.2f%%\ttest error: %.2f%% (%d epochs, %.2f seconds)" %
               (name, (1 - tr_score[1]) * 100, (1 - te_score[1]) * 100, epoch, train_ts))
         initial_epoch = epoch

      del trainer.model
      utils.reset()


   trainers_dict = {}

   for name, trainer in  trainers.items():  
      
      trainers_dict[name] = {'tr_scores': trainer.tr_scores, 'te_scores': trainer.te_scores}    
      
      if name is 'Gauss':
         s = 5
         kernel = lambda x,y: training.Gaussian(x, y, s)
      else:
         s = 10
         kernel = lambda x,y: training.Laplace(x, y, s) 
         
      K = kernel(x_train, x_train)
      alpha_lin = np.linalg.solve(K, y_train)    
      pred_train = K.T.dot(alpha_lin)    
      
      y = y_train
      mse = (np.square(pred_train - y)).mean(axis=None)
      miss = np.count_nonzero(np.argmax(pred_train, axis=1) - np.argmax(y, axis=1)) / y.shape[0]
      trainers_dict[name]['lin_train_mse'] = mse
      trainers_dict[name]['lin_train_ce'] = miss
      
      
      testK = kernel(x_train, x_test)
      pred_test = testK.T.dot(alpha_lin)
      
      y = y_test
      mse = (np.square(pred_test - y)).mean(axis=None)
      miss = np.count_nonzero(np.argmax(pred_test, axis=1) - np.argmax(y, axis=1)) / y.shape[0]  
      trainers_dict[name]['lin_test_mse'] = mse
      trainers_dict[name]['lin_test_ce'] = miss

   trainers_dict

   with open('output/figure1' + dataset + '-' + time.strftime("%Y%m%d-%H%M%S") + '.txt', 'w') as f:
      print(trainers_dict, file=f)

   print()
   print()
   print()
   print(trainers)   
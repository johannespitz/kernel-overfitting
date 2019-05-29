import numpy as np

from keras.datasets.cifar10 import load_data


def unit_range_normalize(X):
   min_ = np.min(X, axis=0)
   max_ = np.max(X, axis=0)
   diff_ = max_ - min_
   diff_[diff_<=0.0] = np.maximum(1.0, min_[diff_<=0.0])
   SX = (X - min_) / diff_
   return SX

def load(grey=True):
   # input image dimensions
   img_rows, img_cols, img_color = 32, 32, 3

   # the data, shuffled and split between train and test sets
   (x_train, y_train), (x_test, y_test) = load_data()

# ( (0.3 * R) + (0.59 * G) + (0.11 * B) ).

   if grey:
      x_train = np.dot(x_train.reshape(x_train.shape[0], img_rows * img_cols, img_color), np.array([0.3, 0.59, 0.11]))
      #x_train[:][:] = x_train[:][:][0] * 0.3 + x_train[:][:][1] * 0.59 + x_train[:][:][2] * 0.11
      x_test = np.dot(x_test.reshape(x_test.shape[0], img_rows * img_cols, img_color), np.array([0.3, 0.59, 0.11]))
      #x_test[:][:] = x_test[:][:][0] * 0.3 + x_test[:][:][1] * 0.59 + x_test[:][:][2] * 0.11
      #print((np.dot(x_train.reshape(x_train.shape[0], img_rows * img_cols, img_color), np.array([0.3, 0.59, 0.11]))).shape)
   else:
      x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols * img_color)
      x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols * img_color)

   x_train = x_train.astype('float32') / 255
   x_test = x_test.astype('float32') / 255

   x_train = unit_range_normalize(x_train)
   x_test = unit_range_normalize(x_test)
   print("Load CIPAR dataset.")
   print(x_train.shape[0], 'train samples')
   print(x_test.shape[0], 'test samples')

   return (x_train, y_train), (x_test, y_test)


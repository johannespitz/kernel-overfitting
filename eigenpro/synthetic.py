import numpy as np


def unit_range_normalize(X):
	min_ = np.min(X, axis=0)
	max_ = np.max(X, axis=0)
	diff_ = max_ - min_
	diff_[diff_<=0.0] = np.maximum(1.0, min_[diff_<=0.0])
	SX = (X - min_) / diff_
	return SX

def load(kind=1):
   train_size = 60000
   test_size = 10000
   dim = 50
   mean1 = 0
   if kind == 1:
      mean2 = 10
   else:
      mean2 = 2

   np.random.seed(1)

   y = np.random.randint(2, size=test_size + train_size)
   mean = np.copy(y)
   mean[mean == 0] = mean1
   mean[mean == 1] = mean2

   x1 = np.random.normal(mean, 1)   

   x = np.column_stack((x1, np.random.uniform(-1, 1, size=(test_size + train_size, dim - 1))))

   # y[y == mean1] = -1
   # y[y == mean2] = 1
   x_train = x[:train_size]
   x_test  = x[train_size:]
   y_train = y[:train_size]
   y_test  = y[train_size:]
   print("Generated Synthetic" + str(kind) + " dataset.")
   print(x_train.shape[0], 'train samples')
   print(x_test.shape[0], 'test samples')

   return (x_train, y_train), (x_test, y_test)

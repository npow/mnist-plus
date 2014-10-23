import h5py
import numpy as np
from sklearn.cross_validation import train_test_split

X = np.load('../blobs/X_train.npy')[:1000]
Y = np.load('../blobs/Y_train.npy')[:1000]
#X_test = np.load('X_test.npy')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

with h5py.File('../blobs/train.h5', 'w') as f:
  f['data'] = X_train
  f['label'] = Y_train.astype(np.float32)
with open('../blobs/train.txt', 'w') as f:
  f.write('examples/hdf5_classification/data/train.h5\n')
  f.write('examples/hdf5_classification/data/train.h5\n')

with h5py.File('../blobs/test.h5', 'w') as f:
  f['data'] = X_test
  f['label'] = Y_test.astype(np.float32)
with open('../blobs/test.txt', 'w') as f:
  f.write('examples/hdf5_classification/data/test.h5\n')
  f.write('examples/hdf5_classification/data/test.h5\n')

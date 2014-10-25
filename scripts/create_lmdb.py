import csv
import h5py
import numpy as np
from sklearn.cross_validation import train_test_split

L = []
with open('../data/train_outputs.csv', 'rb') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  next(reader, None)  # skip the header
  for train_output in reader:  
    id = int(train_output[0])
    klass = int(train_output[1])
    L.append((id, klass))

with open('../data/lmdb_train.txt', 'w') as f:
  for data in L[:35000]:
    f.write('%d.png %d\n' % data)

with open('../data/lmdb_test.txt', 'w') as f:
  for data in L[35000:]:
    f.write('%d.png %d\n' % data)

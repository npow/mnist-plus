import csv
import h5py
import numpy as np
from sklearn.cross_validation import train_test_split

L_train = []
with open('../data/train_outputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_output in reader:  
        id = int(train_output[0])
        klass = int(train_output[1])
        L_train.append((id, klass))

L_test = []
with open('../data/test_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_output in reader:
        id = int(train_output[0])
        klass = -1
        L_test.append((id, klass))

with open('../data/lmdb_train.txt', 'w') as f:
    for data in L_train[:45000]:
        f.write('%d.png %d\n' % data)

with open('../data/lmdb_train_rev.txt', 'w') as f:
    for data in L_train[45000:]:
        f.write('%d.png %d\n' % data)

with open('../data/lmdb_test.txt', 'w') as f:
    for data in L_test:
        f.write('%d.png %d\n' % data)

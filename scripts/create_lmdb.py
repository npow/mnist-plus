import csv
import h5py
import numpy as np
from os import listdir
from sklearn.cross_validation import train_test_split

L_train = []
for f in listdir('../data/blend'):
    z = f.split('.')[0].split(',')
    klass = int(z[1])
    filename = '../blend/%s' % f
    L_train.append((filename, klass))

for f in listdir('../data/composite'):
    z = f.split('.')[0].split(',')
    klass = int(z[1])
    filename = '../composite/%s' % f
    L_train.append((filename, klass))

with open('../data/train_outputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_output in reader:  
        id = int(train_output[0])
        klass = int(train_output[1])
        filename = '../train_images/%d.png' % id
        L_train.append((filename, klass))

L_test = []
with open('../data/test_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_output in reader:
        id = int(train_output[0])
        klass = -1
        filename = '../test_images/%d.png' % id
        L_test.append((filename, klass))

print len(L_train)

with open('../data/lmdb_train.txt', 'w') as f:
    for data in L_train[:60000]:
        f.write('%s %d\n' % data)

with open('../data/lmdb_train_rev.txt', 'w') as f:
    for data in L_train[60000:]:
        f.write('%s %d\n' % data)

with open('../data/lmdb_test.txt', 'w') as f:
    for data in L_test:
        f.write('%s %d\n' % data)

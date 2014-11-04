import os
from PIL import Image
paths = [
    ("data/train_preprocessed/0", "../train_preprocessed/0/"),
#    ("data/train_preprocessed/90", "../train_preprocessed/90/"),
#    ("data/train_preprocessed/180", "../train_preprocessed/180/"),
#    ("data/train_preprocessed/270", "../train_preprocessed/270/"),
    ("data/test_preprocessed/0", "../test_preprocessed/0/"),
#    ("data/test_preprocessed/90", "../test_preprocessed/90/"),
#    ("data/test_preprocessed/180", "../test_preprocessed/180/"),
#    ("data/test_preprocessed/270", "../test_preprocessed/270/")
]

with open('rob_lmdb_train.txt', 'w') as f:
    for p in paths[:1]:
        l = os.listdir(p[0])
        for x in l:
            f.write(p[1] + x + " " + x[x.index(",")+1:x.index(".")] + "\n")

with open('rob_lmdb_test.txt', 'w') as f:
    for p in paths[1:]:
        l = os.listdir(p[0])
        for x in l:
            f.write(p[1] + x + " -1\n")

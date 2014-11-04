import caffe
import numpy as np
import os
import sys
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import *

TRAIN_PATH = 'data/train_preprocessed/0'
TEST_PATH = 'data/test_preprocessed/0'

DUMP_DATA = False
if DUMP_DATA:
    X = []
    Y = []
    for f in os.listdir(TRAIN_PATH):
      image = caffe.io.load_image(os.path.join(TRAIN_PATH, f), color=False)
      image = image.reshape(48*48, 1).T
      X.append(image)
      klass = int(f.split('.')[0].split(',')[1])
      Y.append(klass)
    X = np.vstack(X)
    Y = np.array(Y)
    joblib.dump(X, 'blobs/X_preprocessed.pkl')
    joblib.dump(Y, 'blobs/Y_preprocessed.pkl')
    sys.exit(0)

X = joblib.load('blobs/X_preprocessed.pkl')
Y = joblib.load('blobs/Y_preprocessed.pkl')
print X.shape
print Y.shape


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 2304, 4608, 1152, 576, 10
# 15-20 epochs
clf = LogisticRegression()#DBN([X_train.shape[1], int(sys.argv[1]), int(sys.argv[1]), 10], learn_rates=0.1, learn_rate_decays=0.0, momentum=0.0, epochs=5, verbose=1)
clf.fit(X_train, Y_train)
pred = clf.predict(X_test)

score = metrics.f1_score(Y_test, pred)
print("f1-score:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(Y_test, pred))

print("confusion matrix:")
print(metrics.confusion_matrix(Y_test, pred))

import csv
import numpy as np
import os
import sys
from scipy import linalg
from nolearn.dbn import DBN
from sklearn import grid_search
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA, RandomizedPCA
from sklearn.ensemble import *
from sklearn.externals import joblib
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn import metrics
from sklearn.utils import array2d, as_float_array
from sklearn.base import TransformerMixin, BaseEstimator

class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, regularization=10**-5, copy=False):
        self.regularization = regularization
        self.copy = copy
 
    def fit(self, X, y=None):
        X = array2d(X)
        X = as_float_array(X, copy = self.copy)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        sigma = np.dot(X.T,X) / X.shape[1]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1/np.sqrt(S+self.regularization)))
        self.components_ = np.dot(tmp, U.T)
        return self
 
    def transform(self, X):
        X = array2d(X)
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed, self.components_.T)
        return X_transformed

DUMP_CSV = False
if DUMP_CSV:
    train_inputs = []
    with open('features/trainFeatProcess.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            train_inputs.append(row)
    train_inputs = np.array(train_inputs).T

    test_inputs = []
    with open('features/testFeatProcess.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            test_inputs.append(row)

    test_inputs = np.array(test_inputs).T
    joblib.dump(train_inputs, 'features/X_train_processed.pkl')
    joblib.dump(test_inputs, 'features/X_submit_processed.pkl')
    print train_inputs.shape
    print test_inputs.shape
    sys.exit(0)

X = joblib.load('features/X_train_processed.pkl')
Y = np.load('blobs/Y_train.npy')
print("done loading")

CREATE_SUBMISSION = False
if CREATE_SUBMISSION:
    X_submit = joblib.load('features/X_submit_processed.pkl')
    X = select.transform(X) 
    X_submit = select.transform(X_submit)
    clf = DBN([X.shape[1], 280, 10], learn_rates=.1, learn_rate_decays=0.9, momentum=0.9, epochs=100, verbose=1)
    clf.fit(X, Y)
    print("done fit")
    pred = clf.predict(X_submit)
    f = open('preds_nn_2304x280x10_100epochs_processed.csv', 'wb')
    f.write('Id,Prediction\n')
    for i, p in enumerate(pred):
        f.write("%d,%d\n" % (i+1,p))
    f.close()
else:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    """
    X_train_base, X_train_meta, Y_train_base, Y_train_meta = train_test_split(X_train, Y_train, test_size=0.5, random_state=42)
    X_meta = []
    X_test_meta = []
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf.fit(X_train_base, Y_train_base)
    X_meta.append(rf.predict_proba(X_train_meta))
    X_test_meta.append(rf.predict_proba(X_test))

    svm = LinearSVC()
    svm.fit(X_train_base, Y_train_base)
    X_meta.append(svm.decision_function(X_train_meta))
    X_test_meta.append(svm.decision_function(X_test))
    print("done creating meta")

    X_meta = np.column_stack(X_meta)
    X_test_meta = np.column_stack(X_test_meta)

    clf = LinearSVC()#RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    clf.fit(X_meta, Y_train_meta)
    pred = clf.predict(X_test_meta)
    """

    clf = LinearSVC()
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)

    score = metrics.f1_score(Y_test, pred)
    print("f1-score:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(Y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(Y_test, pred))

import numpy as np
import sys
from nolearn.dbn import DBN
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA, RandomizedPCA
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn import metrics

X = np.load('../blobs/X_train.npy')
Y = np.load('../blobs/Y_train.npy')
print 'done loading'

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def do(selector, k):
    print "K: %d" % k
    if selector is PCA:
        select = selector(n_components=k, whiten=True)
    else:
        select = selector(score_func=chi2, k=k)
    X_train_trunc = select.fit_transform(X_train, Y_train)
    X_test_trunc = select.transform(X_test)

    if k < 100:
        for neighbors in range(1, 5):
            print "kNN: %d" % neighbors
            clf = KNeighborsClassifier(n_neighbors=neighbors)
            clf.fit(X_train_trunc, Y_train)
            pred = clf.predict(X_test_trunc)

            score = metrics.f1_score(Y_test, pred)
            print("f1-score:   %0.3f" % score)

            print("classification report:")
            print(metrics.classification_report(Y_test, pred))

            print("confusion matrix:")
            print(metrics.confusion_matrix(Y_test, pred))
            sys.stdout.flush()

    print "LogisticRegression"
    clf = LogisticRegression()
    clf.fit(X_train_trunc, Y_train)
    pred = clf.predict(X_test_trunc)

    score = metrics.f1_score(Y_test, pred)
    print("f1-score:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(Y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(Y_test, pred))
    sys.stdout.flush()

    #clf = DBN([X_train_trunc.shape[1], X_train_trunc.shape[1], 10], learn_rates=0.3, learn_rate_decays=0.9, epochs=10, verbose=1)
    #clf = LogisticRegression()
    for kernel in ['rbf', 'linear', 'poly']:
        print "SVM kernel: %s" % kernel
        sys.stdout.flush()
        if kernel == 'linear':
            clf = LinearSVC()
        else:
            clf = SVC(kernel=kernel, degree=9)
        clf.fit(X_train_trunc, Y_train)
        pred = clf.predict(X_test_trunc)

        score = metrics.f1_score(Y_test, pred)
        print("f1-score:   %0.3f" % score)

        print("classification report:")
        print(metrics.classification_report(Y_test, pred))

        print("confusion matrix:")
        print(metrics.confusion_matrix(Y_test, pred))
        sys.stdout.flush()


for selector in [SelectKBest, PCA]:
    print "*" * 80
    if selector is PCA:
        print "PCA"
    else:
        print "SelectKBest"
    for k in xrange(10, X_train.shape[1], 500):
        do(selector, k)

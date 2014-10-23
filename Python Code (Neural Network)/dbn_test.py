import numpy as np
from nolearn.dbn import DBN
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA, RandomizedPCA
from sklearn import metrics

X = np.load('../blobs/X_train.npy')
Y = np.load('../blobs/Y_train.npy')
print 'done loading'

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = DBN([X_train.shape[1], X_train.shape[1], 10], learn_rates=0.3, learn_rate_decays=0.9, epochs=10, verbose=1)
clf.fit(X_train, Y_train)
pred = clf.predict(X_test)

score = metrics.f1_score(Y_test, pred)
print("f1-score:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(Y_test, pred))

print("confusion matrix:")
print(metrics.confusion_matrix(Y_test, pred))

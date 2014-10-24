import numpy as np
from nolearn.dbn import DBN
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA, RandomizedPCA
from sklearn.externals import joblib
from sklearn.linear_model import *
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn import grid_search
from sklearn import metrics

X = joblib.load('blobs/o0_l6_X_train.pkl')
Y = joblib.load('blobs/o0_l6_Y_train.pkl')
print 'done loading'
#X = StandardScaler().fit_transform(X)
#X = TruncatedSVD(n_components=1000).fit_transform(X)
print X.shape
print Y.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

#clf = PassiveAggressiveClassifier(C=1.0, fit_intercept=True, loss='hinge', n_iter=50, n_jobs=1, random_state=None, shuffle=False, verbose=0, warm_start=False)
#clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)#SVC(kernel='rbf')
clf = LinearSVC(C=100000000, tol=0.000001)
clf.fit(X_train, Y_train)
pred = clf.predict(X_test)

score = metrics.f1_score(Y_test, pred)
print("f1-score:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(Y_test, pred))

print("confusion matrix:")
print(metrics.confusion_matrix(Y_test, pred))

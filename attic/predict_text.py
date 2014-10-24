import numpy as np
from nolearn.dbn import DBN
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA, RandomizedPCA
from sklearn.externals import joblib
from sklearn.feature_extraction.text import *
from sklearn.linear_model import *
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn import grid_search
from sklearn import metrics

NUM_F = 10

X = joblib.load('blobs/X_klasses.pkl')
Y = joblib.load('blobs/Y_klasses.pkl')
print 'done loading'

def tokenize(s):
  return s[:NUM_F]

vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False)
X = vectorizer.fit_transform(X)
print X.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X_train.todense(), Y_train)
pred = clf.predict(X_test.todense())

score = metrics.f1_score(Y_test, pred)
print("f1-score:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(Y_test, pred))

print("confusion matrix:")
print(metrics.confusion_matrix(Y_test, pred))

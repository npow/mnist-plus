import csv
import overfeat
import numpy as np
from sklearn.externals import joblib
from scipy.ndimage import imread
from scipy.misc import imresize

LAYER = 19
X = None
L = []
with open('data/train_outputs.csv', 'rb') as f:
  reader = csv.reader(f, delimiter=',')
  next(reader, None)
  for data in reader:
    L.append(data)

# quick network: layers 17, 19
# slow network: layers 20, 22
overfeat.init('/Users/npow/code/OverFeat/data/default/net_weight_0', 0)

def extract_features(layer, file_name):
  global X

  image = imread(file_name)
  image.resize((image.shape[0], image.shape[1], 1))

  # overfeat expects rgb, so replicate the grayscale values twice
  image = np.repeat(image, 3, 2)
  image = imresize(image, (231, 231)).astype(np.float32)

  # numpy loads image with colors as last dimension, so transpose tensor
  h = image.shape[0]
  w = image.shape[1]
  c = image.shape[2]
  image = image.reshape(w*h, c)
  image = image.transpose()
  image = image.reshape(c, h, w)

  b = overfeat.fprop(image)
  b = b.flatten()
  x = overfeat.get_output(layer)

  # maxpooling
  x = np.amax(np.amax(x, 1), 1)
  x = x.reshape(x.shape[0], 1).T
  if X is None:
    X = x
  else:
    X = np.append(X, x, axis=0)

Y = []
for data in L[:500]:
  id = data[0]
  klass = int(data[1])
  Y.append(klass)
  extract_features(LAYER, "data/data_as_images/train_images/resized/resized_%s.png" % id)

Y = np.array(Y)
print X.shape
print Y.shape
joblib.dump(X, 'blobs/o1_l%s_X_train.pkl' % LAYER)
joblib.dump(Y, 'blobs/o1_l%s_Y_train.pkl' % LAYER)

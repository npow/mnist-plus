import caffe
import numpy as np
import os

HOME = os.environ['HOME']
MNIST_ROOT = '%s/code/mnist' % HOME
CAFFE_ROOT = '%s/code/caffe' % HOME
MODEL_FILE = '%s/examples/mnist/npow.prototxt' % CAFFE_ROOT
PRETRAINED = '%s/examples/mnist/raw_npow_iter_10000.caffemodel' % CAFFE_ROOT
IMG_PATH = '%s/data/test_images/' % MNIST_ROOT

net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(48,48))
net.set_phase_test()

def classify(id):
  image = caffe.io.load_image('%s/%d.png' % (IMG_PATH, id), color=False)
  p = net.predict([image])
  return p[0].argmax()

def run_test():
  IMG_PATH = '%s/data/train_images/' % MNIST_ROOT
  Y = np.load('blobs/Y_train.npy')
  c = 0
  for id in xrange(1, 50001):
    if id % 100 == 0:
      print id
    klass = classify(id)
    if Y[id-1] != klass:
      c += 1
  print "error: %f" % (c / 50000)

def create_submission():
  f = open('preds_85000.csv', 'wb')
  f.write('Id,Prediction\n')
  for id in xrange(1, 20001):
    if id % 100 == 0:
      print id
    klass = classify(id)
    f.write('%d,%d\n' % (id, klass))
  f.close()

#create_submission()
run_test()

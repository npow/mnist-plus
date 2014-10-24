import numpy as np
import caffe

HOME = '/Users/npow'
MNIST_ROOT = '%s/code/mnist' % HOME
CAFFE_ROOT = '%s/code/caffe' % HOME
MODEL_FILE = '%s/examples/mnist/lenet.prototxt' % CAFFE_ROOT
PRETRAINED = '%s/examples/mnist/npow_iter_1000.caffemodel' % CAFFE_ROOT
IMG_PATH = '%s/data/train_images' % MNIST_ROOT

net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(28,28))
net.set_phase_test()

def classify(id):
  input_image = caffe.io.load_image('%s/%d.png' % (IMG_PATH, id), color=False)
  prediction = net.predict([input_image])
  return prediction[0].argmax()

for id in xrange(1, 10):
  print classify(id)

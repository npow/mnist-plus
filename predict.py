import caffe
import numpy as np
import os
from caffe.proto import caffe_pb2

HOME = os.environ['HOME']
MNIST_ROOT = '%s/code/mnist' % HOME
CAFFE_ROOT = '%s/code/caffe' % HOME
MODEL_FILE = '%s/examples/mnist/raw_lenet.prototxt' % CAFFE_ROOT
PRETRAINED = '%s/examples/mnist/raw_submean_npow_iter_130000.caffemodel' % CAFFE_ROOT
MEAN_FILE = '%s/raw_mean.binaryproto' % CAFFE_ROOT
IMG_PATH = '%s/data/test_images/' % MNIST_ROOT

blob = caffe_pb2.BlobProto()
data = open(MEAN_FILE, 'rb').read()
blob.ParseFromString(data)
IMG_MEAN = caffe.io.blobproto_to_array(blob).reshape(1, 48, 48)

net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=IMG_MEAN, image_dims=(48,48))
net.set_phase_test()

images = [caffe.io.load_image('%s/%d.png' % (IMG_PATH, id), color=False) for id in xrange(1, 20001)]
preds = map(lambda x: x[0].argmax(), net.predict(images))

f = open('preds_123000.csv', 'wb')
f.write('Id,Prediction\n')
for i, klass in enumerate(preds):
  id = i+1
  f.write('%d,%d\n' % (id, klass))
f.close()

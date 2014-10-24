import numpy as np
import caffe

MNIST_ROOT = '/home/ubuntu/code/mnist'
CAFFE_ROOT = '/home/ubuntu/code/caffe'
MODEL_FILE = '%s/examples/mnist/npow.prototxt' % CAFFE_ROOT
PRETRAINED = '/home/ubuntu/npow_iter_20000.caffemodel'
IMG_PATH = '%s/data/train_images' % MNIST_ROOT


net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(48,48))
print 'hi'
print IMG_PATH
net.set_phase_test()
input_image = caffe.io.load_image('%s/1.png' % IMG_DIR)
prediction = net.predict([input_image])
print prediction[0].argmax()

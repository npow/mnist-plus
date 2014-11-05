Machine Learning Digits Classification Project
=====
The dataset is based on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, with modified images. Full description is available [here](http://inclass.kaggle.com/c/difficult-digits)

Implement:  
1. Logistic regression  
2. Feedforward Neural Network   
3. Linear SVM (from [scikit-learn](http://scikit-learn.org))   
4. Convolutional Neural Network ([caffe](https://github.com/npow/caffe))

===
This project uses python 2.7.

```
# pip install numpy
# pip install scikit-learn
# pip install h5py
```

To run the Neural Network code:
```
# cd python && python neural_network.py
```

### Convolutional Neural Network
The architecture used can be found [here](https://github.com/npow/caffe/blob/master/examples/mnist/lenet_train_test.prototxt). To run the cNN:
```
# cd scripts && python create_lmdb.py
# git clone https://github.com/npow/caffe $CAFFE_ROOT
# cd $CAFFE_ROOT
# <follow instructions to build caffe>
# examples/imagenet/npow_imagenet.sh
# examples/mnist/train_lenet.sh
```

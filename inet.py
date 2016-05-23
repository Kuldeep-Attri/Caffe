#import subprocess
#import platform
#import copy

#from sklearn.datasets import load_iris
#import sklearn.metrics 
import numpy as np
#from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt
#%matplotlib inline


plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  

import sys 
sys.path.append("/home/vision/caffe-master/python")
import caffe
import caffe.draw
#import h5py



# Setting Parameters
print "Code is running..."
solver_prototxt_filename = 'solver.prototxt'
train_test_prototxt_filename = 'train_val.prototxt'
deploy_prototxt_filename  = 'deploy.prototxt'
caffemodel_filename = 'bvlc_alexnet.caffemodel'

# This is my network using Alex_Net caffe model. 
net = caffe.Net(deploy_prototxt_filename,caffemodel_filename, caffe.TEST)
	
mu = np.load('/home/vision/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(50,        # batch size
						  3,         # 3-channel (BGR) images
						  227, 227)  # image size is 227x227



image = caffe.io.load_image('/home/vision/caffe-master/examples/images/00002.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()

# load ImageNet labels
labels_file = '/home/vision/caffe-master/data/ilsvrc12/synset_words.txt'
		
labels = np.loadtxt(labels_file, str, delimiter='\t')
print 'output label:', labels[output_prob.argmax()]
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/marta/prj/caffe/python')
import caffe
import os

net = caffe.Net('../models/deploy.prototxt', caffe.TEST)
caffe.set_mode_cpu()
# if we have a GPU, we wiil also write
# caffe.set_device(0)
# caffe.set_mode_gpu()

print '\nnet.inputs = ', net.inputs
print '\ndir(net.blobs) = ', dir(net.blobs)   # net.blobs for data in layers
print '\ndir(net.params) = ', dir(net.params)   # net.params for weights and biases in layers
print '\nconv shape = ', net.blobs['conv'].data.shape  # net.blobs['data'] contains an array of shape (1,3,48,48)
# The first '1' refers to the number of images
# the second '1' refers to the number of channels in the image
# the third and forth refer to the shape of the image

print 'net.params[conv][0] = ', net.params['conv'][0]  # contains the weight parameters of neurons
print 'net.params[conv][1] = ', net.params['conv'][1]  # contains the bias parameters of neurons


img = cv2.imread('../resource/buildings.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img = img.swapaxes(0, 2).swapaxes(1, 2)
img_blob_inp = img[np.newaxis, :, :, :]

net.blobs['data'].reshape(*img_blob_inp.shape)
cv2.cvtColor(img_blob_inp, cv2.COLOR_GRAY2RGB)
net.blobs['data'].data[...] = img_blob_inp

net.forward()

for i in xrange(10):
    cv2.imwrite('../resource/output_img_' + str(i) + '.jpg', net.blobs['conv'].data[0, i])

net.save('../models/network.caffemodel')    # to save layer in order not to go through the process again

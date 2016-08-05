import caffe
import numpy as np

# loading the model into the 'net' object
net = caffe.Net('/home/marta/PycharmProjects/learning_caffe/models/deploy.prototxt',
'/home/marta/PycharmProjects/learning_caffe/results/_iter_10000.caffemodel', caffe.TEST)

# to preprocess the input image and transform it into something that Caffe can understand
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('/home/marta/prj/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2, 0, 1))        # to transform an image from (256,256,3) to (3,256,256)
transformer.set_channel_swap('data', (2, 1, 0))     # if using RGB instead of BGR
transformer.set_raw_scale('data', 255.0)        # to normalize the values in the image based on the 0 - 255
net.blobs['data'].reshape(1, 3, 48, 48)
# loading the input image
img_new = caffe.io.load_image('../resource/oleg.jpg')

net.blobs['data'].data[...] = transformer.preprocess('data', img_new)

output = net.forward()

print output['prob'].argmax()   # predicted output class

# # printing all the predicted labels:
# label_mapping = np.loadtxt("../resource/class.txt", str, delimiter='\t')
# best_n = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
# print label_mapping[best_n]

'''
This file contains a keras Regularizer that can be used to encourage the largest sigular
value (lsv) of either a Dense or a Conv2D layer to be <= 1.

For Dense layers the linear transformation they perform is a multiplication by the
weight matrix W (and then the addition of a bias term, but this does not affect the gradient
so we can ignore it). So, for a Dense layer the lsv is just: lsv(W)

For Conv2D layers things are more complicated. The weight matrix W does define the
transformation, but it isn't defined simply as multiplication of the input by W, rather,
the input is multiplied by a matrix that is some function of W (and the size of the
input image, which we will call im_sz). Lets call the suitable function frmr, then for
a Conv2D layer the lsv is: lsv(frmr(W,im_sz)).

(As an asside, in the spectral normalization paper they seem to directly use lsv(W) even
in the Conv2D case. I think this is technically wrong, but has a sort of similar result,
in that it does provide some pressure for lsv(frmr(W,im_sz)) to be low-ish, but making
lsv(W) <= 1 doesn't neccessarily imply lsv(frmr(W,im_sz)) <= 1)

We need to define both lsv and frmr in a differentiable way so they can be used to
train a network with backprop. The spectral normalization paper proposes a method to
approximate lsv in a differentiable way (which we will come back to later), so we just
need to work out a method for frmr.

Lets assume that the Conv2D layer has padding and a stride of 1, and also assume the
input is a single channel image, and we only have 1 filter. This means that the layer
maps (im_sz, im_sz, 1) --> (im_sz, im_sz, 1). Lets define n = im_sz * imsz, then
the layer defines a linear map from n-dimensional space to n-dimensional space. So, we
can flatten the image to an n-dimensional vector, then multiply it by some n by n
matrix, M, to get the n-dimensional vector output, which can be then reshaped into the
output image. Thus M = frmr(W,im_sz) (and so lsv(M) = lsv(frmr(W,im_sz)) ).

Specifically, M is made by arranging (and duplicating) the values of W into an n by n
matrix. we do this in the function make_M().

frmr(W,im_sz) = f(W)*g(im_sz)

a,b,c,d
e,f,g,h
i,j,k,l
m,n,o,p

'''

from keras import backend as K
from keras.regularizers import Regularizer
import numpy as np


def my_im2col(img, W, pad=True, stride=1):
    # matrix will map from w*h*num_channels to (w/stride)*(h/stride)*num_filters

    w, h, num_channels = img.shape
    filter_width, filter_height, _, num_filters = W.shape

    if pad:
        w_pad, h_pad = 0, 0
    else:
        w_pad, h_pad = filter_width / 2, filter_height / 2

    M = np.zeros((((w - w_pad * 2) / stride) * ((h - h_pad * 2) / stride) * num_filters, w * h * num_channels))

    # print '--'
    # print M.shape
    # print W.shape

    row = 0
    for filter in range(num_filters):
        for y_pos in range(h_pad, h - h_pad, stride):
            for x_pos in range(w_pad, w - w_pad, stride):
                for channel in range(num_channels):
                    ind = x_pos + w * y_pos + w * h * channel
                    for fx in range(-(filter_width / 2), (filter_width + 1) / 2):
                        for fy in range(-(filter_height / 2), (filter_height + 1) / 2):
                            if (0 <= x_pos + fx < w) and (0 <= y_pos + fy < h):
                                # print row, ind+fx+fy*w
                                # print (filter_width/2)+fx,(filter_height/2)+fy,channel,filter
                                M[row, ind + fx + fy * w] = W[
                                    (filter_width / 2) + fx, (filter_height / 2) + fy, channel, filter]
                row += 1

    return M


def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = numpy.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


def my_im2col_np(img, W, pad=True, stride=1):
    w, h, num_channels = img.shape
    filter_width, filter_height, _, num_filters = W.shape

    if pad:
        w_pad, h_pad = 0, 0
    else:
        w_pad, h_pad = filter_width / 2, filter_height / 2

    M = np.zeros((((w - w_pad * 2) / stride) * ((h - h_pad * 2) / stride) * num_filters, w * h * num_channels))

    indexes = []
    values = []
    for channel in range(num_channels):
        ind = x_pos + w * y_pos + w * h * channel
        for fx in range(-(filter_width / 2), (filter_width + 1) / 2):
            for fy in range(-(filter_height / 2), (filter_height + 1) / 2):
                if (0 <= x_pos + fx < w) and (0 <= y_pos + fy < h):
                    M[row, ind + fx + fy * w] = W[(filter_width / 2) + fx, (filter_height / 2) + fy, channel, filter]

    return M


def largestSingularValues(l, imsz=3):
    '''
    This function takes a keras model as an input and returns a list of the
    largest singular values of each of its weights matricies.

    Useful for sanity checking.
    '''

    layer_type = str(type(l)).split('.')[-1][:-2]

    if layer_type == 'Model':
        SVs = []
        for sub_l in l.layers:
            if len(sub_l.get_weights()):
                SVs = SVs + largestSingularValues(sub_l, imsz)
        return SVs

    elif layer_type == 'Dense':
        W = l.get_weights()[0]
        # W = np.reshape(W, (-1,W.shape[-1]))
        _, s, _ = np.linalg.svd(W)
        return [s[0]]

    elif layer_type == 'Conv2D':
        # note, this is only an approximation. I think it's a lower bound?
        W = l.get_weights()[0]
        img = np.zeros((imsz, imsz, W.shape[2]))
        M = my_im2col(img, W)
        _, s, _ = np.linalg.svd(M)
        return [s[0]]

    else:
        return []


def largestSingularValues_old(l):
    '''
    This function takes a keras model as an input and returns a list of the
    largest singular values of each of its weights matricies.

    Useful for sanity checking.
    '''

    layer_type = str(type(l)).split('.')[-1][:-2]

    if layer_type == 'Model':
        SVs = []
        for sub_l in l.layers:
            if len(sub_l.get_weights()):
                SVs = SVs + largestSingularValues_old(sub_l)
        return SVs

    elif layer_type == 'Dense' or layer_type == 'Conv2D':
        W = l.get_weights()[0]
        W = np.reshape(W, (-1, W.shape[-1]))
        _, s, _ = np.linalg.svd(W)
        return [s[0]]


# def largestSingularValues(model):
# 	'''
# 	This function takes a keras model as an input and returns a list of the
# 	largest singular values of each of its weights matricies.

# 	Useful for sanity checking.
# 	'''

# 	SVs = []
# 	for l in model.layers:
# 		if len(l.get_weights()):

# 			layer_type = str(type(l)).split('.')[-1][:-2]

# 			if layer_type == 'Dense':
# 				W = l.get_weights()[0]
# 				W = np.reshape(W, (-1,W.shape[-1]))
# 				_, s, _ = np.linalg.svd(W)
# 				SVs.append(s[0])

# 			if layer_type == 'Conv2D':
# 				pass

# 	return SVs

class Spectral(Regularizer):
    ''' Spectral normalization regularizer
        # Arguments
            alpha = weight for regularization penalty
    '''

    def __init__(self, dim, alpha=K.variable(10.)):
        '''
        in a Conv2D layer dim needs to be num_channels in the previous layer times the filter_size^2
        in a Dense layer dim needs to be num_channels in the previous layer
        '''

        self.dim = dim
        self.alpha = alpha  # K.cast_to_floatx(alpha)
        self.u = K.variable(np.random.random((dim, 1)) * 2 - 1.)

    def __call__(self, x):
        # return K.mean(K.abs(x))

        # print K.int_shape(x)

        x_shape = K.shape(x)
        x = K.reshape(x, (-1, x_shape[-1]))  # this deals with convolutions, fingers crossed!
        # x = K.transpose(K.reshape(x, (-1, x_shape[-1]))) #this deals with convolutions, fingers crossed!

        # print K.int_shape(x)
        # print K.shape(self.u)

        # self.u = K.variable(np.random.random((self.dim,1))*2-1.)

        for itters in range(3):
            WTu = K.dot(K.transpose(x), self.u)
            v = WTu / K.sqrt(K.sum(K.square(WTu)))

            Wv = K.dot(x, v)
            self.u = Wv / K.sqrt(K.sum(K.square(Wv)))

        spectral_norm = K.dot(K.dot(K.transpose(self.u), x), v)

        target_x = K.stop_gradient(x / spectral_norm)
        return self.alpha * K.mean(K.abs(target_x - x))

        # return self.alpha * K.switch(K.greater(spectral_norm, 1), spectral_norm, 0*spectral_norm)

        return self.alpha * K.abs(1 - spectral_norm)  # + 0.3 * K.sum(K.abs(x))

    def get_config(self):
        return {'alpha': float(self.alpha)}


if __name__ == '__main__':

    from matplotlib import pyplot as plt

    # M[row, ind+fx+fy*w] = W[(filter_width/2)+fx,(filter_height/2)+fy,channel,filter]

    # row = 0
    # for filter in range(num_filters):
    # 	for y_pos in range(h_pad,h-h_pad,stride):
    # 		for x_pos in range(w_pad,w-w_pad,stride):
    # 			for channel in range(num_channels):
    # 				ind = x_pos + w*y_pos + w*h*channel
    # 				for fx in range(-(filter_width/2), (filter_width+1)/2):
    # 					for fy in range(-(filter_height/2), (filter_height+1)/2):
    # 						if (0 <= x_pos + fx < w) and (0 <= y_pos + fy < h):
    # 							# print row, ind+fx+fy*w
    # 							# print (filter_width/2)+fx,(filter_height/2)+fy,channel,filter
    # 							M[row, ind+fx+fy*w] = W[(filter_width/2)+fx,(filter_height/2)+fy,channel,filter]
    # 			row += 1


    # stride = 1
    # im_w, im_h = 5, 5
    # filter_width, filter_height, channels, filters = 3, 3, 2, 1
    # w_pad, h_pad = 0, 0

    # W = np.ones((filter_width, filter_height, channels, filters))
    # M = np.zeros((((im_w-w_pad*2)/stride)*((im_h-h_pad*2)/stride)*filters, im_w*im_h*channels))

    # i1 = np.array(range(filter_width))
    # i2 = np.array(range(filter_height))
    # i3 = np.array(range(channels))
    # i4 = np.array(range(filters))

    # M[i1, i2*3, i3*im_w*im_h] = W[i1, i2, i3]

    # print M
    # sys.exit()

    im_sz = 3
    channels = 1
    filter_sz = 3

    W = np.random.randn(filter_sz, filter_sz, channels, 1)

    img = np.random.randn(1, im_sz, im_sz, channels)
    M = my_im2col(img[0], W)
    base = np.linalg.svd(M)[1][0]

    previous = base

    img3 = np.random.randn(1, 3, 3, channels)
    img7 = np.random.randn(1, 7, 7, channels)

    X, Y = [], []
    for i in range(100):
        W = np.random.randn(filter_sz, filter_sz, channels, 1)

        M = my_im2col(img3[0], W)
        lsv_3 = np.linalg.svd(M)[1][0]

        M = my_im2col(img7[0], W)
        lsv_7 = np.linalg.svd(M)[1][0]

        X.append(lsv_3)
        Y.append(lsv_7 / lsv_3)

    plt.scatter(X, Y)
    plt.show()

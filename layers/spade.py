import tensorflow as tf
from keras.engine import Layer
from keras.layers import LeakyReLU, Conv2D, Add, Lambda
from keras_contrib.layers import InstanceNormalization


def spade_block(conf, anatomy_input, layer, fin, fout):
    learn_shortcut = (fin != fout)
    fmiddle = min(fin, fout)

    l1 = _spade(conf, anatomy_input, layer, fin)
    l2 = LeakyReLU(0.2)(l1)
    l3 = Conv2D(fmiddle, 3, padding='same')(l2)

    l4 = _spade(conf, anatomy_input, l3, fmiddle)
    l5 = LeakyReLU(0.2)(l4)
    l6 = Conv2D(fout, 3, padding='same')(l5)

    if learn_shortcut:
        layer = _spade(conf, anatomy_input, layer, fin)
        layer = Conv2D(fout, 1, padding='same', use_bias=False)(layer)

    return Add()([layer, l6])


def _spade(conf, anatomy_input, layer, f):
    layer = InstanceNormalization(scale=False, center=False)(layer)
    anatomy = Lambda(resize_like, arguments={'ref_tensor': layer})(anatomy_input)
    anatomy = Conv2D(128, 3, padding='same', activation='relu')(anatomy)
    gamma = Conv2D(f, 3, padding='same')(anatomy)
    beta  = Conv2D(f, 3, padding='same')(anatomy)
    return SPADE_COND()([layer, gamma, beta])
    # return Add()([Multiply()([layer, gamma]), beta])


def resize_like(input_tensor, ref_tensor): # resizes input tensor wrt. ref_tensor
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [H.value, W.value])


class SPADE_COND(Layer):
    '''
    The SPADE conditioning
    '''

    def __init__(self, **kwargs):
        super(SPADE_COND, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SPADE_COND, self).build(input_shape)

    def call(self, x, **kwargs):
        x, gamma, beta = x

        return x * (1 + gamma) + beta

    def compute_output_shape(self, input_shape):
        return input_shape
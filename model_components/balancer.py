
import logging

import keras.backend as K
import tensorflow as tf
from keras import Input, Model
from keras.layers import Concatenate, Dense, Lambda

log = logging.getLogger('pair_selector')

def build(conf):
    """
    Build a model that predicts similarity weights between anatomies. These are based on the overlap of the anatomies
    calculated with Dice
    :param conf: a configuration object
    """
    x1 = Input(shape=conf.anatomy_encoder.output_shape)
    x2 = Input(shape=conf.anatomy_encoder.output_shape)
    x3 = Input(shape=conf.anatomy_encoder.output_shape)
    x4 = Input(shape=conf.anatomy_encoder.output_shape)

    overlap = [Lambda(dice)([x1, x]) for x in [x2, x3, x4]]
    x = Concatenate()(overlap)
    l = Dense(5, activation='relu')(x)
    w = Dense(conf.n_pairs, name='beta')(l)
    w = Lambda(lambda x: tf.nn.softmax(x, dim=-1))(w)

    m = Model(inputs=[x1, x2, x3, x4], outputs=w, name='Balancer')
    log.info('Balancer')
    m.summary(print_fn=log.info)
    return m

def dice(y):
    y_true, y_pred = y
    intersection = K.sum(y_true * y_pred, axis=(1, 2, 3))
    union = K.sum(y_true, axis=(1, 2, 3)) + K.sum(y_pred, axis=(1, 2, 3))
    dice  = (2 * intersection + 1e-12) / (union + 1e-12)
    return K.expand_dims(dice, axis=1)
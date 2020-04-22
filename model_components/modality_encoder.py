import logging

from keras import Input, Model
from keras.layers import Concatenate, Conv2D, LeakyReLU, Flatten, Dense, \
    Lambda

import costs
from utils.sdnet_utils import sampling

log = logging.getLogger('modality_encoder')


def build(conf):
    """
    Build an encoder to extract intensity information from the image.
    :param conf: a configuration object
    """
    anatomy = Input(conf.anatomy_encoder.output_shape)
    image   = Input(conf.input_shape)

    z_mean, z_log_var = build_simple_encoder(conf, anatomy, image)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, name='z')([z_mean, z_log_var])
    divergence = Lambda(costs.kl, name='divergence')([z_mean, z_log_var])

    model = Model(inputs=[anatomy, image], outputs=[z, divergence], name='Enc_Modality')
    log.info('Enc_Modality')
    model.summary(print_fn=log.info)
    return model


def build_simple_encoder(conf, anatomy, image):
    l = Concatenate(axis=-1)([anatomy, image])
    l = Conv2D(16, 3, strides=2, kernel_initializer='he_normal')(l)
    l = LeakyReLU()(l)
    l = Conv2D(32, 3, strides=2, kernel_initializer='he_normal')(l)
    l = LeakyReLU()(l)
    l = Conv2D(64, 3, strides=2, kernel_initializer='he_normal')(l)
    l = LeakyReLU()(l)
    l = Conv2D(128, 3, strides=2, kernel_initializer='he_normal')(l)
    l = LeakyReLU()(l)

    l = Flatten()(l)
    l = Dense(32, kernel_initializer='he_normal')(l)
    l = LeakyReLU()(l)

    z_mean    = Dense(conf.num_z, name='z_mean')(l)
    z_log_var = Dense(conf.num_z, name='z_log_var')(l)

    return z_mean, z_log_var

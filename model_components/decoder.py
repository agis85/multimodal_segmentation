import logging

from keras import Input, Model
from keras.layers import Dense, LeakyReLU, Reshape, Conv2D, Add, UpSampling2D

from layers.film import FiLM
from layers.spade import spade_block

log = logging.getLogger('decoder')


def build(conf):
    """
    Build a decoder that generates an image by combining an anatomical and a modality
    representation. Two decoders are considered based on FiLM or SPADE conditioning.
    :param conf: a configuration object
    """
    anatomy_input  = Input(shape=conf.anatomy_encoder.output_shape)
    modality_input = Input((conf.num_z,))

    if conf.decoder_type == 'film':
        l = _film_decoder(anatomy_input, modality_input)
    elif conf.decoder_type == 'spade':
        l = _spade_decoder(conf, anatomy_input, modality_input)
    else:
        raise ValueError('Unknown decoder_type value: ' + str(conf.decoder_type))

    l = Conv2D(1, 1, activation='tanh', padding='same', kernel_initializer='glorot_normal')(l)
    log.info('Decoder')

    model = Model(inputs=[anatomy_input, modality_input], outputs=l, name='Decoder')
    model.summary(print_fn=log.info)
    return model


def _gamma_beta_pred(inp, num_chn):
    gamma = Dense(num_chn)(inp)
    gamma = LeakyReLU()(gamma)
    beta = Dense(num_chn)(inp)
    beta = LeakyReLU()(beta)
    return gamma, beta


def _film_layer(anatomy_input, modality_input):
    l1 = Conv2D(8, 3, padding='same')(anatomy_input)
    l1 = LeakyReLU()(l1)

    l2 = Conv2D(8, 3, strides=1, padding='same')(l1)
    gamma_l2, beta_l2 = _gamma_beta_pred(modality_input, 8)
    l2 = FiLM()([l2, gamma_l2, beta_l2])
    l2 = LeakyReLU()(l2)

    l = Add()([l1, l2])
    return l


def _film_decoder(anatomy_input, modality_input):
    l  = Conv2D(8, 3, padding='same')(anatomy_input)
    l  = LeakyReLU()(l)
    l1 = _film_layer(l, modality_input)
    l2 = _film_layer(l1, modality_input)
    l3 = _film_layer(l2, modality_input)
    l4 = _film_layer(l3, modality_input)
    return l4


def _spade_decoder(conf, anatomy_input, modality_input):
    modality = Dense(conf.input_shape[0] * conf.input_shape[1] * 128 // 1024)(modality_input)
    l1 = Reshape((conf.input_shape[0] // 32, conf.input_shape[1] // 32, 128))(modality)
    l1 = spade_block(conf, anatomy_input, l1, 128, 128)
    l1 = UpSampling2D(size=2)(l1)
    l1 = spade_block(conf, anatomy_input, l1, 128, 128)
    l1 = UpSampling2D(size=2)(l1)
    l2 = spade_block(conf, anatomy_input, l1, 128, 128)
    l3 = UpSampling2D(size=2)(l2)
    l4 = spade_block(conf, anatomy_input, l3, 128, 64)
    l5 = UpSampling2D(size=2)(l4)
    l6 = spade_block(conf, anatomy_input, l5, 64, 32)
    l7 = UpSampling2D(size=2)(l6)
    l8 = spade_block(conf, anatomy_input, l7, 32, 16)
    return l8

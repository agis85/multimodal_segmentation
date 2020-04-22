import logging

from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation

log = logging.getLogger('segmentor')


def build(conf):
    """
    Build a segmentation network that converts anatomical maps to segmentation masks.
    :param conf: a configuration object
    """
    inp = Input(conf.anatomy_encoder.output_shape)

    l = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inp)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)

    # +1 output for background
    output = Conv2D(conf.num_masks + 1, 1, padding='same', activation='softmax')(l)

    model = Model(inputs=inp, outputs=output, name='Segmentor')
    log.info('Segmentor')
    model.summary(print_fn=log.info)
    return model

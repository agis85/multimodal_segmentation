import logging

import keras.layers
from keras import Input, Model

from layers import stn_spline
from layers.stn_spline import ThinPlateSpline2D

log = logging.getLogger('anatomy_fuser')


def build(conf):
    """
    Build a model that deforms and fuses anatomies, used to combine multimodal information.
    Two anatomies are assumed: the first anatomy is deformed to match the second.
    Deformation model uses a STN.
    :param conf: a configuration object
    """
    anatomy1 = Input(conf.anatomy_encoder.output_shape)  # anatomy from modality1
    anatomy2 = Input(conf.anatomy_encoder.output_shape)  # anatomy from modality2

    output_shape = conf.anatomy_encoder.output_shape

    dims = conf.anatomy_encoder.output_shape[:-1]
    cp = [5, 5]
    channels = conf.anatomy_encoder.out_channels

    locnet = stn_spline.build_locnet(output_shape, output_shape, cp[0] * cp[1] * 2)
    theta = locnet([anatomy1, anatomy2])
    anatomy1_deformed = ThinPlateSpline2D(dims, cp, channels)([anatomy1, theta])

    # Fusion step
    anatomy_fused = keras.layers.Maximum()([anatomy1_deformed, anatomy2])

    model = Model(inputs=[anatomy1, anatomy2], outputs=[anatomy1_deformed, anatomy_fused], name='Anatomy_Fuser')
    log.info('Anatomy fuser')
    model.summary(print_fn=log.info)
    return model

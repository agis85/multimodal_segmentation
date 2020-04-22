from keras.layers import BatchNormalization, Lambda, Conv2D, LeakyReLU, UpSampling2D, Activation

from keras_contrib.layers import InstanceNormalization


def normalise(norm=None, **kwargs):
    if norm == 'instance':
        return InstanceNormalization(**kwargs)
    elif norm == 'batch':
        return BatchNormalization()
    else:
        return Lambda(lambda x : x)


def upsample_block(l0, f, norm_name, activation='relu'):
    l = UpSampling2D(size=2)(l0)
    l = Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(l)
    l = normalise(norm_name)(l)

    if activation == 'leakyrelu':
        return LeakyReLU()(l)
    return Activation(activation)(l)



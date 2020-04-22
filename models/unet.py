# UNet Implementation of 4 downsampling and 4 upsampling blocks.
# Each block has 2 convolutions, batch normalisation, relu and a residual connection.
# The number of filters for the 1st layer is 64 and at every block, this is doubled. Each upsampling blocks halves the
# number of filters.


from keras import Input, Model
from keras.layers import Concatenate, Conv2D, MaxPooling2D, Activation

from models.basenet import BaseNet
from utils.model_utils import upsample_block, normalise
import logging
log = logging.getLogger('unet')


class UNet(BaseNet):
    def __init__(self, conf):
        super(UNet, self).__init__(conf)
        self.input_shape  = conf.input_shape
        self.out_channels = conf.out_channels

        self.normalise        = conf.normalise
        self.f                = conf.filters
        self.downsample       = conf.downsample
        assert self.downsample > 0, 'Unet downsample must be over 0.'

    def build(self):
        self.input = Input(shape=self.input_shape)
        l = self.unet_downsample(self.input, self.normalise)
        self.unet_bottleneck(l, self.normalise)
        l = self.unet_upsample(self.bottleneck, self.normalise)
        out = self.out(l)
        self.model = Model(inputs=self.input, outputs=out)
        self.model.summary(print_fn=log.info)
        self.load_models()

    def unet_downsample(self, inp, normalise):
        self.d_l0 = conv_block(inp, self.f, normalise)
        l = MaxPooling2D(pool_size=(2, 2))(self.d_l0)

        if self.downsample > 1:
            self.d_l1 = conv_block(l, self.f * 2, normalise)
            l = MaxPooling2D(pool_size=(2, 2))(self.d_l1)

        if self.downsample > 2:
            self.d_l2 = conv_block(l, self.f * 4, normalise)
            l = MaxPooling2D(pool_size=(2, 2))(self.d_l2)

        if self.downsample > 3:
            self.d_l3 = conv_block(l, self.f * 8, normalise)
            l = MaxPooling2D(pool_size=(2, 2))(self.d_l3)
        return l

    def unet_bottleneck(self, l, normalise, name=''):
        flt = self.f * 2
        if self.downsample > 1:
            flt *= 2
        if self.downsample > 2:
            flt *= 2
        if self.downsample > 3:
            flt *= 2
        self.bottleneck = conv_block(l, flt, normalise, name)
        return self.bottleneck

    def unet_upsample(self, l, normalise):
        if self.downsample > 3:
            l = upsample_block(l, self.f * 8, normalise, activation='linear')
            l = Concatenate()([l, self.d_l3])
            l = conv_block(l, self.f * 8, normalise)

        if self.downsample > 2:
            l = upsample_block(l, self.f * 4, normalise, activation='linear')
            l = Concatenate()([l, self.d_l2])
            l = conv_block(l, self.f * 4, normalise)

        if self.downsample > 1:
            l = upsample_block(l, self.f * 2, normalise, activation='linear')
            l = Concatenate()([l, self.d_l1])
            l = conv_block(l, self.f * 2, normalise)

        if self.downsample > 0:
            l = upsample_block(l, self.f, normalise, activation='linear')
            l = Concatenate()([l, self.d_l0])
            l = conv_block(l, self.f, normalise)

        return l

    def out(self, l, out_activ=None):
        if out_activ is None:
            out_activ = 'sigmoid' if self.out_channels == 1 else 'softmax'
        return Conv2D(self.out_channels, 1, padding='same', activation=out_activ)(l)


def conv_block(l0, f, norm_name, name=''):
    l = Conv2D(f, 3, strides=1, padding='same', kernel_initializer='he_normal')(l0)
    l = normalise(norm_name)(l)
    l = Activation('relu')(l)

    l = Conv2D(f, 3, strides=1, padding='same', kernel_initializer='he_normal')(l)
    l = normalise(norm_name)(l)
    return Activation('relu', name=name)(l)

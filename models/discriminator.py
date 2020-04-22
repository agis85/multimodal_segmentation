from keras import Input, Model
from keras.layers import Conv2D, LeakyReLU, Flatten, Dense
from keras.optimizers import Adam

from layers.spectralnorm import Spectral
from models.basenet import BaseNet


class Discriminator(BaseNet):
    '''
    DCGAN Discriminator with Spectral Norm and LS-GAN loss
    '''
    def __init__(self, conf):
        super(Discriminator, self).__init__(conf)

    def build(self):
        inp_shape = self.conf.input_shape
        name      = self.conf.name
        f         = self.conf.filters
        downsample_blocks = 3 if not hasattr(self.conf, 'downsample_blocks') else self.conf.downsample_blocks
        assert downsample_blocks > 1, downsample_blocks

        d_input = Input(inp_shape)
        l = Conv2D(f, 4, strides=2, kernel_initializer="he_normal")(d_input)
        l = LeakyReLU(0.2)(l)

        for i in range(downsample_blocks):
            s = 1 if i == downsample_blocks - 1 else 2
            spectral_params = f * (2 ** i)
            l = self._downsample_block(l, f * 2 * (2 ** i), s, spectral_params)

        l = Flatten()(l)
        l = Dense(1, activation="linear")(l)

        self.model = Model(d_input, l, name=name)
        return self.model

    def _downsample_block(self, l0, f, stride, spectral_params, name=''):
        l = Conv2D(f, 4, strides=stride, kernel_initializer="he_normal",
                   kernel_regularizer=Spectral( spectral_params * 4 * 4, 10.), name=name)(l0)
        return LeakyReLU(0.2)(l)

    def compile(self):
        assert self.model is not None, 'Model has not been built'
        self.model.compile(optimizer=Adam(lr=self.conf.lr), loss='mse')

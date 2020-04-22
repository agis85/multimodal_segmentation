import logging

from keras import Input, Model
from keras.layers import Conv2D, Activation, UpSampling2D, Concatenate

from layers.rounding import Rounding
from models.unet import UNet
from utils.model_utils import normalise

log = logging.getLogger('anatomy_encoder')


def build(conf, name='Enc_Anatomy'):
    """
    Build a UNet based encoder to extract anatomical information from the image.
    """
    spatial_encoder = UNet(conf)
    spatial_encoder.input = Input(shape=conf.input_shape)
    l1_down = spatial_encoder.unet_downsample(spatial_encoder.input, spatial_encoder.normalise)    # downsample
    spatial_encoder.unet_bottleneck(l1_down, spatial_encoder.normalise)                            # bottleneck
    l2_up   = spatial_encoder.unet_upsample(spatial_encoder.bottleneck, spatial_encoder.normalise) # upsample

    anatomy = Conv2D(conf.out_channels, 1, padding='same', activation='softmax', name='conv_anatomy')(l2_up)
    if conf.rounding:
        anatomy = Rounding()(anatomy)

    model = Model(inputs=spatial_encoder.input, outputs=anatomy, name=name)
    log.info('Enc_Anatomy')
    model.summary(print_fn=log.info)
    return model

class AnatomyEncoders(object):

    def __init__(self, modalities):
        self.modalities = modalities

    def build(self, conf):
        # build encoder1
        encoder1 = UNet(conf)
        encoder1.input = Input(shape=conf.input_shape)
        l1 = encoder1.unet_downsample(encoder1.input, encoder1.normalise)

        # build encoder2
        encoder2 = UNet(conf)
        encoder2.input = Input(shape=conf.input_shape)
        l2 = encoder2.unet_downsample(encoder2.input, encoder2.normalise)

        self.build_decoder(conf)

        d1_l3 = encoder1.d_l3 if conf.downsample > 3 else None
        d2_l3 = encoder2.d_l3 if conf.downsample > 3 else None
        anatomy_output1 = self.evaluate_decoder(conf,
                                                l1, d1_l3, encoder1.d_l2, encoder1.d_l1, encoder1.d_l0)
        anatomy_output2 = self.evaluate_decoder(conf,
                                                l2, d2_l3, encoder2.d_l2, encoder2.d_l1, encoder2.d_l0)

        # build shared layer
        shr_lay4 = Conv2D(conf.out_channels, 1, padding='same', activation='softmax', name='conv_anatomy')

        # connect models
        encoder1_output = shr_lay4(anatomy_output1)
        encoder2_output = shr_lay4(anatomy_output2)

        if conf.rounding:
            encoder1_output = Rounding()(encoder1_output)
            encoder2_output = Rounding()(encoder2_output)

        encoder1 = Model(inputs=encoder1.input, outputs=encoder1_output,
                         name='Enc_Anatomy_%s' % self.modalities[0])
        encoder2 = Model(inputs=encoder2.input, outputs=encoder2_output,
                         name='Enc_Anatomy_%s' % self.modalities[1])

        return [encoder1, encoder2]

    def evaluate_decoder(self, conf, decoder_input, decoder_l3, decoder_l2, decoder_l1, decoder_l0):
        l0_out = self.l0_6(self.l0_5(self.l0_4(self.l0_3(self.l0_2(self.l0_1(decoder_input))))))

        if conf.downsample > 3:
            l3_out = self.l3(self.l2(self.l1(l0_out)))
            l4_out = self.l4([l3_out, decoder_l3])

            l10_out = self.l10(self.l9(self.l8(self.l7(self.l6(self.l5(l4_out))))))
        else:
            l10_out = l0_out

        l13_out = self.l13(self.l12(self.l11(l10_out)))
        l14_out = self.l14([l13_out, decoder_l2])

        l20_out = self.l20(self.l19(self.l18(self.l17(self.l16(self.l15(l14_out))))))
        l23_out = self.l23(self.l22(self.l21(l20_out)))
        l24_out = self.l14([l23_out, decoder_l1])

        l30_out = self.l30(self.l29(self.l28(self.l27(self.l26(self.l25(l24_out))))))
        l33_out = self.l33(self.l32(self.l31(l30_out)))
        l34_out = self.l34([l33_out, decoder_l0])

        l40_out = self.l40(self.l39(self.l38(self.l37(self.l36(self.l35(l34_out))))))
        return l40_out

    def build_decoder(self, conf):
        f0 = conf.filters * 16 if conf.downsample > 3 else conf.filters * 8
        self.l0_1 = Conv2D(f0, 3, padding='same', kernel_initializer='he_normal')
        self.l0_2 = normalise(conf.normalise)
        self.l0_3 = Activation('relu')
        self.l0_4 = Conv2D(f0, 3, padding='same', kernel_initializer='he_normal')
        self.l0_5 = normalise(conf.normalise)
        self.l0_6 = Activation('relu')

        if conf.downsample > 3:
            self.l1 = UpSampling2D(size=2)
            self.l2 = Conv2D(conf.filters * 8, 3, padding='same', kernel_initializer='he_normal')
            self.l3 = normalise(conf.normalise)

            self.l4 = Concatenate()
            self.l5 = Conv2D(conf.filters * 8, 3, strides=1, padding='same', kernel_initializer='he_normal')
            self.l6 = normalise(conf.normalise)
            self.l7 = Activation('relu')
            self.l8 = Conv2D(conf.filters * 8, 3, strides=1, padding='same', kernel_initializer='he_normal')
            self.l9 = normalise(conf.normalise)
            self.l10 = Activation('relu')

        self.l11 = UpSampling2D(size=2)
        self.l12 = Conv2D(conf.filters * 4, 3, padding='same', kernel_initializer='he_normal')
        self.l13 = normalise(conf.normalise)

        self.l14 = Concatenate()
        self.l15 = Conv2D(conf.filters * 4, 3, strides=1, padding='same', kernel_initializer='he_normal')
        self.l16 = normalise(conf.normalise)
        self.l17 = Activation('relu')
        self.l18 = Conv2D(conf.filters * 4, 3, strides=1, padding='same', kernel_initializer='he_normal')
        self.l19 = normalise(conf.normalise)
        self.l20 = Activation('relu')

        self.l21 = UpSampling2D(size=2)
        self.l22 = Conv2D(conf.filters * 2, 3, padding='same', kernel_initializer='he_normal')
        self.l23 = normalise(conf.normalise)

        self.l24 = Concatenate()
        self.l25 = Conv2D(conf.filters * 2, 3, strides=1, padding='same', kernel_initializer='he_normal')
        self.l26 = normalise(conf.normalise)
        self.l27 = Activation('relu')
        self.l28 = Conv2D(conf.filters * 2, 3, strides=1, padding='same', kernel_initializer='he_normal')
        self.l29 = normalise(conf.normalise)
        self.l30 = Activation('relu')

        self.l31 = UpSampling2D(size=2)
        self.l32 = Conv2D(conf.filters, 3, padding='same', kernel_initializer='he_normal')
        self.l33 = normalise(conf.normalise)

        self.l34 = Concatenate()
        self.l35 = Conv2D(conf.filters, 3, strides=1, padding='same', kernel_initializer='he_normal')
        self.l36 = normalise(conf.normalise)
        self.l37 = Activation('relu')
        self.l38 = Conv2D(conf.filters, 3, strides=1, padding='same', kernel_initializer='he_normal')
        self.l39 = normalise(conf.normalise)
        self.l40 = Activation('relu')
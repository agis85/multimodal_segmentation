
import logging
import os
import numpy as np
from keras import Input, Model
from keras.layers import Lambda
from keras.optimizers import Adam

import costs
from model_components import anatomy_encoder, anatomy_fuser, modality_encoder, segmentor, decoder
from models.basenet import BaseNet
from models.discriminator import Discriminator
from utils.sdnet_utils import make_trainable, get_net

log = logging.getLogger('mmsdnet')


class MMSDNet(BaseNet):
    def __init__(self, conf):
        super(MMSDNet, self).__init__(conf)

        self.modalities = conf.modality # list of input modalities

        self.D_Mask           = None  # Mask Discriminator
        self.Encoders_Anatomy = None  # list of anatomy encoders for every modality
        self.Enc_Modality     = None  # Modality Encoder
        self.Enc_Modality_mu  = None  # The mean value of the Modality Encoder prediction
        self.Anatomy_Fuser    = None  # Anatomy Fuser that deforms and fused anatomies
        self.Segmentor        = None  # Segmentation network
        self.Decoder          = None  # Decoder network

        self.D_Mask_trainer       = None  # Trainer for mask discriminator
        self.unsupervised_trainer = None  # Trainer when having unlabelled data
        self.supervised_trainer   = None  # Trainer when using data with labels.
        self.Z_Regressor          = None  # Trainer for reconstructing a sampled Z

    def build(self):
        self.build_mask_discriminator()
        self.build_generators()
        self.load_models()

    def load_models(self):
        if os.path.exists(self.conf.folder + '/supervised_trainer'):
            log.info('Loading trained models from file')

            self.supervised_trainer.load_weights(self.conf.folder + '/supervised_trainer')

            self.Encoders_Anatomy = [get_net(self.supervised_trainer, 'Enc_Anatomy_%s' % mod) for mod in self.modalities]

            self.Enc_Modality = get_net(self.supervised_trainer, 'Enc_Modality')
            self.Enc_Modality_mu = Model(self.Enc_Modality.inputs, self.Enc_Modality.get_layer('z_mean').output)
            self.Anatomy_Fuser= get_net(self.supervised_trainer, 'Anatomy_Fuser')
            self.Segmentor    = get_net(self.supervised_trainer, 'Segmentor')
            self.Decoder      = get_net(self.supervised_trainer, 'Decoder')
            self.D_Mask       = get_net(self.supervised_trainer, 'D_Mask')
            self.build_z_regressor()

    def save_models(self):
        log.debug('Saving trained models')
        self.supervised_trainer.save_weights(self.conf.folder + '/supervised_trainer')

    def build_mask_discriminator(self):
        # Build a discriminator for masks.
        D = Discriminator(self.conf.d_mask_params)
        D.build()
        log.info('Mask Discriminator D_M')
        D.model.summary(print_fn=log.info)
        self.D_Mask = D.model

        real_M = Input(self.conf.d_mask_params.input_shape)
        fake_M = Input(self.conf.d_mask_params.input_shape)
        real = self.D_Mask(real_M)
        fake = self.D_Mask(fake_M)

        self.D_Mask_trainer = Model([real_M, fake_M], [real, fake], name='D_Mask_trainer')
        self.D_Mask_trainer.compile(Adam(lr=self.conf.d_mask_params.lr), loss='mse')
        self.D_Mask_trainer.summary(print_fn=log.info)

    def build_generators(self):
        assert self.D_Mask is not None, 'Discriminator has not been built yet'
        make_trainable(self.D_Mask_trainer, False)

        self.Encoders_Anatomy = [anatomy_encoder.build(self.conf.anatomy_encoder, 'Enc_Anatomy_%s' % mod)
                                 for mod in self.modalities]
        self.Anatomy_Fuser    = anatomy_fuser.build(self.conf)
        self.Enc_Modality     = modality_encoder.build(self.conf)
        self.Enc_Modality_mu  = Model(self.Enc_Modality.inputs, self.Enc_Modality.get_layer('z_mean').output)
        self.Segmentor        = segmentor.build(self.conf)
        self.Decoder          = decoder.build(self.conf)

        self.build_unsupervised_trainer()  # build standard gan for data with no labels
        self.build_supervised_trainer()
        self.build_z_regressor()

    def build_unsupervised_trainer(self):
        # Model for unsupervised training

        # inputs
        x_list = [Input(shape=self.conf.input_shape) for _ in self.modalities]
        num_mod = len(self.modalities)

        # x -> s, z -> m, y
        s_list = [self.Encoders_Anatomy[i](x_list[i]) for i in range(num_mod)]
        z_list = [self.Enc_Modality([s_list[i], x_list[i]]) for i in range(num_mod)]
        m1, m2 = self.Segmentor(s_list[0]), self.Segmentor(s_list[1])

        m_list = [m1]
        adv_m_list  = [self.D_Mask(Lambda(lambda x: x[..., 0:self.conf.num_masks])(m)) for m in [m1, m2]]
        rec_x_list  = [self.Decoder([s_list[i], z_list[i][0]]) for i in range(num_mod)]

        # segment deformed and fused
        s1_def, s1_fused = self.Anatomy_Fuser(s_list)
        s2_def, s2_fused = self.Anatomy_Fuser(list(reversed(s_list)))

        fused_segmentations = [self.Segmentor(s) for s in [s1_def, s1_fused, s2_def, s2_fused]]
        m_list += fused_segmentations[2:] # there are masks only for modality1
        adv_m_list  += [self.D_Mask(Lambda(lambda x: x[..., 0:self.conf.num_masks])(s)) for s in fused_segmentations]

        # reconstruct deformed and fused
        z_list_s1def = [self.Enc_Modality([s, x_list[1]]) for s in [s1_def, s1_fused]]
        rec_x_list  += [self.Decoder([s, z_list_s1def[i][0]]) for i, s in enumerate([s1_def, s1_fused])]

        z_list_s2def = [self.Enc_Modality([s, x_list[0]]) for s in [s2_def, s2_fused]]
        rec_x_list  += [self.Decoder([s, z_list_s2def[i][0]]) for i, s in enumerate([s2_def, s2_fused])]

        # list of KL divergences for every modality
        diverg_list  = [z_list[i][1] for i in range(num_mod)]
        diverg_list += [z_list_s1def[i][1] for i in range(num_mod)]
        diverg_list += [z_list_s2def[i][1] for i in range(num_mod)]

        all_outputs = m_list + adv_m_list + rec_x_list + diverg_list
        self.unsupervised_trainer = Model(inputs=x_list, outputs=all_outputs)
        log.info('Unsupervised trainer')
        self.unsupervised_trainer.summary(print_fn=log.info)

        loss_list = [costs.make_dice_loss_fnc(self.loader.num_masks) for _ in range(3)] + \
                    ['mse'] * (num_mod * 3) + \
                    ['mae'] * (num_mod * 3) + \
                    [costs.ypred for _ in range(num_mod * 3)]
        weights_list = [self.conf.w_sup_M for _ in range(3)] + \
                       [self.conf.w_adv_M for _ in range(num_mod * 3)] + \
                       [self.conf.w_rec_X for _ in range(num_mod * 3)] + \
                       [self.conf.w_kl    for _ in range(num_mod * 3)]
        self.unsupervised_trainer.compile(Adam(self.conf.lr), loss=loss_list, loss_weights=weights_list)

    def build_supervised_trainer(self):
        # Model for unsupervised training

        # inputs
        x_list = [Input(shape=self.conf.input_shape) for _ in self.modalities]
        num_mod = len(self.modalities)

        s_list = [self.Encoders_Anatomy[i](x_list[i]) for i in range(num_mod)]
        z_list = [self.Enc_Modality([s_list[i], x_list[i]]) for i in range(num_mod)]
        m_list = [self.Segmentor(s) for s in s_list]
        adv_m_list  = [self.D_Mask(Lambda(lambda x: x[..., 0:self.conf.num_masks])(m)) for m in m_list]
        rec_x_list  = [self.Decoder([s_list[i], z_list[i][0]]) for i in range(num_mod)]

        # segment deformed and fused
        s1_def, s1_fused = self.Anatomy_Fuser(s_list)
        s2_def, s2_fused = self.Anatomy_Fuser(list(reversed(s_list)))

        fused_segmentations = [self.Segmentor(s) for s in [s1_def, s1_fused, s2_def, s2_fused]]
        m_list      += fused_segmentations
        adv_m_list  += [self.D_Mask(Lambda(lambda x: x[..., 0:self.conf.num_masks])(s)) for s in fused_segmentations]

        # reconstruct deformed and fused
        z_list_s1def = [self.Enc_Modality([s, x_list[1]]) for s in [s1_def, s1_fused]]
        rec_x_list  += [self.Decoder([s, z_list_s1def[i][0]]) for i, s in enumerate([s1_def, s1_fused])]

        z_list_s2def = [self.Enc_Modality([s, x_list[0]]) for s in [s2_def, s2_fused]]
        rec_x_list  += [self.Decoder([s, z_list_s2def[i][0]]) for i, s in enumerate([s2_def, s2_fused])]

        # list of KL divergences for every modality
        diverg_list  = [z_list[i][1] for i in range(num_mod)]
        diverg_list += [z_list_s1def[i][1] for i in range(num_mod)]
        diverg_list += [z_list_s2def[i][1] for i in range(num_mod)]

        all_outputs = m_list + adv_m_list + rec_x_list + diverg_list
        self.supervised_trainer = Model(inputs=x_list, outputs=all_outputs)
        log.info('Supervised trainer')
        self.supervised_trainer.summary(print_fn=log.info)

        loss_list = [costs.make_dice_loss_fnc(self.loader.num_masks) for _ in range(num_mod * 3)] + \
                    ['mse'] * (num_mod * 3) + \
                    ['mae'] * (num_mod * 3) + \
                    [costs.ypred for _ in range(num_mod * 3)]
        weights_list = [self.conf.w_sup_M for _ in range(num_mod * 3)] + \
                       [self.conf.w_adv_M for _ in range(num_mod * 3)] + \
                       [self.conf.w_rec_X for _ in range(num_mod * 3)] + \
                       [self.conf.w_kl    for _ in range(num_mod * 3)]
        self.supervised_trainer.compile(Adam(self.conf.lr), loss=loss_list, loss_weights=weights_list)

    def build_z_regressor(self):
        num_inputs = len(self.modalities) + 4
        s_list = [Input(self.conf.anatomy_encoder.output_shape) for _ in range(num_inputs)]
        sample_z_list = [Input((self.conf.num_z,)) for _ in range(num_inputs)]
        sample_x_list = [self.Decoder([s_list[i], sample_z_list[i]]) for i in range(num_inputs)]

        rec_Z_list = [self.Enc_Modality_mu([s_list[i], sample_x_list[i]]) for i in range(num_inputs)]

        all_inputs = s_list + sample_z_list
        self.Z_Regressor = Model(inputs=all_inputs, outputs=rec_Z_list, name='ZReconstruct')
        log.info('Z Regressor')
        self.Z_Regressor.summary(print_fn=log.info)
        losses  = ['mae'] * (num_inputs)
        weights = [self.conf.w_rec_Z for _ in range(num_inputs)]
        self.Z_Regressor.compile(Adam(self.conf.lr), loss=losses, loss_weights=weights)

    def predict_mask(self, modality_index, type, image_list):
        assert type in ['simple', 'def', 'max', 'maxnostn']

        idx2 = modality_index
        idx1 = 1 - idx2

        images_mod1 = image_list[idx1]
        images_mod2 = image_list[idx2]

        s1 = self.Encoders_Anatomy[idx1].predict(images_mod1)
        s2 = self.Encoders_Anatomy[idx2].predict(images_mod2)

        if type == 'simple':
            return self.Segmentor.predict(s2)
        elif type == 'def':
            return self.Segmentor.predict(self.Anatomy_Fuser.predict([s1, s2])[0])
        elif type == 'max':
            return self.Segmentor.predict(self.Anatomy_Fuser.predict([s1, s2])[1])
        elif type == 'maxnostn':
            s_max_nostn = np.max([s1, s2], axis=0)
            return self.Segmentor.predict(s_max_nostn)

        raise ValueError(type)
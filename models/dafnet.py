import logging
import traceback

from keras import Input, Model
from keras.layers import Lambda, Multiply, Add
from keras.optimizers import Adam

import costs
from model_components import anatomy_fuser, modality_encoder, segmentor, decoder, balancer
from model_components.anatomy_encoder import AnatomyEncoders
from models.discriminator import Discriminator
from models.mmsdnet import MMSDNet
from utils.sdnet_utils import make_trainable

log = logging.getLogger('dafnet')


class DAFNet(MMSDNet):
    def __init__(self, conf):
        super(DAFNet, self).__init__(conf)

        self.D_Mask           = None  # Mask Discriminator
        self.D_Image1         = None  # Image Discriminator for modality 1
        self.D_Image2         = None  # Image Discriminator for modality 2
        self.Encoders_Anatomy = None  # list of anatomy encoders for every modality
        self.Enc_Modality     = None  # Modality Encoder
        self.Enc_Modality_mu  = None  # The mean value of the Modality Encoder prediction
        self.Anatomy_Fuser    = None  # Anatomy Fuser that deforms and fused anatomies
        self.Segmentor        = None  # Segmentation network
        self.Decoder          = None  # Decoder network
        self.Balancer         = None  # Model that calculates weighs similarity of anatomies

        # Trainers
        self.D_Mask_trainer       = None  # Trainer for mask discriminator
        self.D_Image1_trainer     = None  # Trainer for image modality1 discriminator
        self.D_Image2_trainer     = None  # Trainer for image modality2 discriminator
        self.unsupervised_trainer = None  # Trainer when having unlabelled data
        self.supervised_trainer   = None  # Trainer when using data with labels.
        self.Z_Regressor          = None  # Trainer for reconstructing a sampled Z

    def build(self):
        self.build_mask_discriminator()
        self.build_image_discriminator1()
        self.build_image_discriminator2()

        self.build_generators()
        try:
            self.load_models()
        except:
            log.warning('No models found')
            traceback.print_exc()
            pass

    def load_models(self):
        log.info('Loading trained models from file')

        model_folder = self.conf.folder + '/models/'
        self.D_Mask.load_weights(model_folder + '/D_Mask')
        self.D_Image1.load_weights(model_folder + '/D_Image1')
        self.D_Image2.load_weights(model_folder + '/D_Image2')

        self.Encoders_Anatomy[0].load_weights(model_folder + 'Enc_Anatomy1')
        self.Encoders_Anatomy[1].load_weights(model_folder + 'Enc_Anatomy2')
        self.Enc_Modality.load_weights(model_folder + 'Enc_Modality')
        self.Anatomy_Fuser.load_weights(model_folder + 'Anatomy_Fuser')
        self.Segmentor.load_weights(model_folder + 'Segmentor')
        self.Decoder.load_weights(model_folder + 'Decoder')
        try:
            self.Balancer.load_weights(model_folder + 'Balancer')
        except:
            pass

        self.build_trainers()

    def build_image_discriminator1(self):
        """
        Build a discriminator for images
        """
        params1 = self.conf.d_image_params
        params1['name'] = 'D_Image1'
        D = Discriminator(params1)
        D.build()
        log.info('Image Discriminator D_I')
        D.model.summary(print_fn=log.info)
        self.D_Image1 = D.model

        real_x = Input(self.conf.d_image_params.input_shape)
        fake_x = Input(self.conf.d_image_params.input_shape)
        real = self.D_Image1(real_x)
        fake = self.D_Image1(fake_x)

        self.D_Image1_trainer = Model([real_x, fake_x], [real, fake], name='D_Image1_trainer')
        self.D_Image1_trainer.compile(Adam(lr=self.conf.d_image_params.lr), loss='mse')
        self.D_Image1_trainer.summary(print_fn=log.info)

    def build_image_discriminator2(self):
        """
        Build a discriminator for images
        """
        params2 = self.conf.d_image_params
        params2['name'] = 'D_Image2'
        D = Discriminator(params2)
        D.build()
        log.info('Image Discriminator D_I2')
        D.model.summary(print_fn=log.info)
        self.D_Image2 = D.model

        real_x = Input(self.conf.d_image_params.input_shape)
        fake_x = Input(self.conf.d_image_params.input_shape)
        real = self.D_Image2(real_x)
        fake = self.D_Image2(fake_x)

        self.D_Image2_trainer = Model([real_x, fake_x], [real, fake], name='D_Image2_trainer')
        self.D_Image2_trainer.compile(Adam(lr=self.conf.d_image_params.lr), loss='mse')
        self.D_Image2_trainer.summary(print_fn=log.info)

    def build_generators(self):
        assert self.D_Mask is not None, 'Discriminator has not been built yet'
        make_trainable(self.D_Mask, False)
        make_trainable(self.D_Image1, False)
        make_trainable(self.D_Image2, False)

        self.Encoders_Anatomy = AnatomyEncoders(self.modalities).build(self.conf.anatomy_encoder)
        self.Anatomy_Fuser    = anatomy_fuser.build(self.conf)
        self.Enc_Modality     = modality_encoder.build(self.conf)
        self.Enc_Modality_mu  = Model(self.Enc_Modality.inputs, self.Enc_Modality.get_layer('z_mean').output)
        self.Segmentor = segmentor.build(self.conf)
        self.Decoder   = decoder.build(self.conf)
        self.Balancer  = balancer.build(self.conf)

        self.build_trainers()

    def build_trainers(self):
        self.build_z_regressor()
        if not self.conf.automatedpairing:
            self.build_trainers_expertpairs()
        else:
            self.build_trainers_automatedpairs()

    def build_trainers_expertpairs(self):
        """
        Build trainer models for unsupervised and supervised learning, when multimodal data are expertly paired.
        This assumes two modalities
        """
        losses = {'Segmentor': costs.make_combined_dice_bce(self.loader.num_masks), 'D_Mask': 'mse', 'Decoder': 'mae',
                  'D_Image1': 'mse', 'D_Image2': 'mse', 'Enc_Modality': costs.ypred, 'ZReconstruct': 'mae'}
        loss_weights = {'Segmentor': self.conf.w_sup_M, 'D_Mask': self.conf.w_adv_M, 'Decoder': self.conf.w_rec_X,
                        'D_Image1': self.conf.w_adv_X, 'D_Image2': self.conf.w_adv_X, 'Enc_Modality': self.conf.w_kl,
                        'ZReconstruct': self.conf.w_rec_Z}

        all_inputs, all_outputs = self.get_params_expert_pairing(supervised=False)
        self.unsupervised_trainer = Model(inputs=all_inputs, outputs=all_outputs)
        log.info('Unsupervised model trainer')
        self.unsupervised_trainer.summary(print_fn=log.info)
        self.unsupervised_trainer.compile(Adam(self.conf.lr), loss=losses, loss_weights=loss_weights)

        all_inputs, all_outputs = self.get_params_expert_pairing(supervised=True)
        self.supervised_trainer = Model(inputs=all_inputs, outputs=all_outputs)
        log.info('Supervised model trainer')
        self.supervised_trainer.summary(print_fn=log.info)
        self.supervised_trainer.compile(Adam(self.conf.lr), loss=losses, loss_weights=loss_weights)

    def get_params_expert_pairing(self, supervised):
        """
        Connect the DAFNet components for supervised or unsupervised training
        :return: a list of inputs and outputs
        """
        # inputs
        x1 = Input(shape=self.conf.input_shape)
        x2 = Input(shape=self.conf.input_shape)

        # encode
        s1 = self.Encoders_Anatomy[0](x1)
        s2 = self.Encoders_Anatomy[1](x2)
        z1, kl1 = self.Enc_Modality([s1, x1])
        z2, kl2 = self.Enc_Modality([s2, x2])

        # segment
        m1 = self.Segmentor(s1)
        m2 = self.Segmentor(s2)

        # decoder
        y1 = self.Decoder([s1, z1])
        y2 = self.Decoder([s2, z2])

        # GANs
        adv_m1 = self.D_Mask(Lambda(lambda x: x[..., 0:self.conf.num_masks])(m1))
        adv_m2 = self.D_Mask(Lambda(lambda x: x[..., 0:self.conf.num_masks])(m2))
        adv_y1 = self.D_Image1(y1)
        adv_y2 = self.D_Image2(y2)

        # deform and fuse
        s1_def, _ = self.Anatomy_Fuser([s1, s2])
        s2_def, _ = self.Anatomy_Fuser([s2, s1])

        # segment
        m2_s1_def = self.Segmentor(s1_def)
        m1_s2_def = self.Segmentor(s2_def)

        # decoder (cross-reconstruction)
        y2_s1_def = self.Decoder([s1_def, z2])
        y1_s2_def = self.Decoder([s2_def, z1])

        # GANs
        adv_m2_s1_def = self.D_Mask(Lambda(lambda x: x[..., 0:self.conf.num_masks])(m2_s1_def))
        adv_m1_s2_def = self.D_Mask(Lambda(lambda x: x[..., 0:self.conf.num_masks])(m1_s2_def))
        adv_y2_s1_def = self.D_Image2(y2_s1_def)
        adv_y1_s2_def = self.D_Image1(y1_s2_def)

        # Z-Regressor
        z1_input = Input(shape=(self.conf.num_z,))
        z2_input = Input(shape=(self.conf.num_z,))
        [z1_rec, z2_rec] = self.Z_Regressor([s1, s2, z1_input, z2_input])

        # inputs / outputs
        all_inputs = [x1, x2, z1_input, z2_input]
        all_outputs  = [m1, m2, m1_s2_def, m2_s1_def] if supervised else [m1, m1_s2_def]
        all_outputs += [adv_m1, adv_m2, adv_m1_s2_def, adv_m2_s1_def] + \
                       [y1, y2, y1_s2_def, y2_s1_def] + \
                       [adv_y1, adv_y2, adv_y1_s2_def, adv_y2_s1_def] + \
                       [kl1, kl2, z1_rec, z2_rec]
        return all_inputs, all_outputs

    def build_trainers_automatedpairs(self):
        """
        Build trainer models for unsupervised and supervised learning, when multimodal data are automatically paired.
        This assumes two modalities
        """
        losses = {'Segmentor': costs.make_combined_dice_bce(self.loader.num_masks), 'SegmentorDef': costs.ypred,
                  'D_Mask': 'mse', 'Decoder': 'mae', 'DecoderDef': costs.ypred,
                  'D_Image1': 'mse', 'D_Image2': 'mse', 'Enc_Modality': costs.ypred, 'ZReconstruct': 'mae'}
        loss_weights = {'Segmentor': self.conf.w_sup_M, 'SegmentorDef': self.conf.w_sup_M, 'D_Mask': self.conf.w_adv_M,
                        'Decoder': self.conf.w_rec_X, 'DecoderDef': self.conf.w_rec_X, 'D_Image1': self.conf.w_adv_X,
                        'D_Image2': self.conf.w_adv_X, 'Enc_Modality': self.conf.w_kl, 'ZReconstruct': self.conf.w_rec_Z}

        all_inputs, all_outputs = self.get_params_automated_pairing(supervised=False)
        self.unsupervised_trainer = Model(inputs=all_inputs, outputs=all_outputs)
        log.info('Unupervised model trainer')
        self.unsupervised_trainer.summary(print_fn=log.info)
        self.unsupervised_trainer.compile(Adam(self.conf.lr), loss=losses, loss_weights=loss_weights)

        all_inputs, all_outputs = self.get_params_automated_pairing(supervised=True)
        self.supervised_trainer = Model(inputs=all_inputs, outputs=all_outputs)
        log.info('Supervised model trainer')
        self.supervised_trainer.summary(print_fn=log.info)
        self.supervised_trainer.compile(Adam(self.conf.lr), loss=losses, loss_weights=loss_weights)

    def get_params_automated_pairing(self, supervised):
        """
        Connect the DAFNet components for supervised or unsupervised training
        :return: a list of inputs and outputs
        """
        # inputs
        x1_lst   = [Input(shape=self.conf.input_shape) for _ in range(self.conf.n_pairs)]
        x2_lst   = [Input(shape=self.conf.input_shape) for _ in range(self.conf.n_pairs)]
        m1_input = Input(shape=self.conf.input_shape[:-1] + [self.conf.num_masks + 1])
        x1 = x1_lst[0]
        x2 = x2_lst[0]

        # encode
        s1_lst = [self.Encoders_Anatomy[0](x) for x in x1_lst]
        s2_lst = [self.Encoders_Anatomy[1](x) for x in x2_lst]
        s1 = s1_lst[0]
        s2 = s2_lst[0]
        z1, kl1 = self.Enc_Modality([s1, x1])
        z2, kl2 = self.Enc_Modality([s2, x2])

        # segment
        m1 = self.Segmentor(s1)
        m2 = self.Segmentor(s2)

        # decode
        y1 = self.Decoder([s1, z1])
        y2 = self.Decoder([s2, z2])

        # GANs
        adv_m1 = self.D_Mask(Lambda(lambda x: x[..., 0:self.conf.num_masks])(m1))
        adv_m2 = self.D_Mask(Lambda(lambda x: x[..., 0:self.conf.num_masks])(m2))
        adv_y1 = self.D_Image1(y1)
        adv_y2 = self.D_Image2(y2)

        # deform and fuse
        s1_def_lst = [self.Anatomy_Fuser([s1_i, s2])[0] for s1_i in s1_lst]
        w1_def_lst = self.calculate_weights([s2] + s1_def_lst)

        s2_def_lst = [self.Anatomy_Fuser([s2_i, s1])[0] for s2_i in s2_lst]
        w2_def_lst = self.calculate_weights([s1] + s2_def_lst)

        # decoder (cross-reconstruction)
        DecoderLoss = Lambda(lambda x: costs.mae_single_input(x))
        DecoderDef = Add(name='DecoderDef')

        y2_s1_def_lst = [self.Decoder([s1_def, z2]) for s1_def in s1_def_lst]
        y1_s2_def_lst = [self.Decoder([s2_def, z1]) for s2_def in s2_def_lst]
        y2_s1_def = DecoderDef([Multiply()([w, DecoderLoss([x2, y2_s1_def])])
                                for w, y2_s1_def in zip(w1_def_lst, y2_s1_def_lst)])
        y1_s2_def = DecoderDef([Multiply()([w, DecoderLoss([x1, y1_s2_def])])
                                for w, y1_s2_def in zip(w2_def_lst, y1_s2_def_lst)])

        # segment
        SegmentorDef = Add(name='SegmentorDef')
        SegmentorLoss = Lambda(lambda x: costs.make_combined_dice_bce_perbatch(self.loader.num_masks)(x[0], x[1]))

        m1_s2_def_lst = [self.Segmentor(s2_def) for s2_def in s2_def_lst]
        m1_s2_def     = SegmentorDef([Multiply()([w, SegmentorLoss([m1_input, m1_s2_def])])
                                      for w, m1_s2_def in zip(w2_def_lst, m1_s2_def_lst)])
        m2_s1_def_lst = [self.Segmentor(s1_def) for s1_def in s1_def_lst]

        if supervised:
            m2_input = Input(shape=self.conf.input_shape[:-1] + [self.conf.num_masks + 1])
            m2_s1_def = SegmentorDef([Multiply()([w, SegmentorLoss([m2_input, m2_s1_def])])
                                      for w, m2_s1_def in zip(w1_def_lst, m2_s1_def_lst)])

        # GANs
        adv_m2_s1_def = self.D_Mask(Lambda(lambda x: x[..., 0:self.conf.num_masks])(m2_s1_def_lst[0]))
        adv_m1_s2_def = self.D_Mask(Lambda(lambda x: x[..., 0:self.conf.num_masks])(m1_s2_def_lst[0]))
        adv_y2_s1_def = self.D_Image2(y2_s1_def_lst[0])
        adv_y1_s2_def = self.D_Image1(y1_s2_def_lst[0])

        # Z-Regressor
        z1_input = Input(shape=(self.conf.num_z,))
        z2_input = Input(shape=(self.conf.num_z,))
        [z1_rec, z2_rec] = self.Z_Regressor([s1, s2, z1_input, z2_input])

        # outputs
        all_inputs   = x1_lst + x2_lst + [m1_input, m2_input, z1_input, z2_input] if supervised \
            else x1_lst + x2_lst + [m1_input, z1_input, z2_input]
        all_outputs  = [m1, m2, m1_s2_def, m2_s1_def] if supervised else [m1, m1_s2_def]
        all_outputs += [adv_m1, adv_m2, adv_m1_s2_def, adv_m2_s1_def] + \
                       [y1, y2, y1_s2_def, y2_s1_def] + \
                       [adv_y1, adv_y2, adv_y1_s2_def, adv_y2_s1_def] + \
                       [kl1, kl2, z1_rec, z2_rec]

        return all_inputs, all_outputs

    def build_z_regressor(self):
        """
        Regress the modality factor. Assumes 4 inputs: 2 s-factors for the 2 modalities
        """
        num_inputs = 2

        s_lst = [Input(self.conf.anatomy_encoder.output_shape) for _ in range(num_inputs)]
        z_lst = [Input((self.conf.num_z,)) for _ in range(num_inputs)]
        y_lst = [self.Decoder([s, z]) for s, z in zip(s_lst, z_lst)]

        z_rec_lst = [self.Enc_Modality_mu([s, y]) for s, y in zip(s_lst, y_lst)]

        self.Z_Regressor = Model(inputs=s_lst + z_lst, outputs=z_rec_lst, name='ZReconstruct')
        self.Z_Regressor.compile(Adam(self.conf.lr), loss=['mae', 'mae'],
                                 loss_weights=[self.conf.w_rec_Z, self.conf.w_rec_Z])

    def calculate_weights(self, inputs):
        s_mod2 = inputs[0]
        s_list = inputs[1:]

        if len(s_list) == 1:
            return None

        weights = self.Balancer([s_mod2] + s_list)
        weights = [Lambda(lambda x: x[..., j:j+1])(weights) for j in range(self.conf.n_pairs)]
        return weights

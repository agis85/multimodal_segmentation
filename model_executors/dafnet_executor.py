import logging
import os

import numpy as np
from keras.callbacks import CSVLogger, EarlyStopping
from keras.utils import Progbar

import costs
import utils.data_utils
import utils.image_utils
from callbacks.dafnet_image_callback import DAFNetImageCallback
from callbacks.loss_callback import SaveLoss
from callbacks.swa import SWA
from model_components import anatomy_encoder, modality_encoder, anatomy_fuser, segmentor, decoder, balancer
from model_executors.base_executor import Executor
from models.discriminator import Discriminator
from utils.distributions import NormalDistribution

log = logging.getLogger('dafnet_executor')


class DAFNetExecutor(Executor):
    """
    Train a DAFNet or MMSDNet model using parameters stored in the configuration.
    """
    def __init__(self, conf, model):
        super(DAFNetExecutor, self).__init__(conf, model)
        self.exp_clb = None
        self.loader.modalities = self.conf.modality

        self.gen_labelled = None  # iterator for labelled data (supervised learning)
        self.gen_unlabelled      = None  # iterator for unlabelled data (unsupervised learning)
        self.discriminator_masks = None  # iterator for real masks to train discriminators
        self.discriminator_image = None  # iterator for images to train discriminators
        self.img_callback = None         # callback to save images
        self.data = None                 # labelled data container of type MultimodalPairedData
        self.ul_data = None              # unlabelled data container of type MultimodalPairedData

        self.init_swa_models()

    def init_swa_models(self):
        """
        Initialise objects for Stochastic Weight Averaging
        """
        self.swa_D_Mask        = SWA(20, Discriminator(self.conf.d_mask_params).build, None)
        self.swa_D_Image1      = SWA(20, Discriminator(self.conf.d_image_params).build, None)
        self.swa_D_Image2      = SWA(20, Discriminator(self.conf.d_image_params).build, None)
        self.swa_Enc_Anatomy1  = SWA(20, anatomy_encoder.build, self.conf.anatomy_encoder)
        self.swa_Enc_Anatomy2  = SWA(20, anatomy_encoder.build, self.conf.anatomy_encoder)
        self.swa_Enc_Modality  = SWA(20, modality_encoder.build, self.conf)
        self.swa_Anatomy_Fuser = SWA(20, anatomy_fuser.build, self.conf)
        self.swa_Segmentor     = SWA(20, segmentor.build, self.conf)
        self.swa_Decoder       = SWA(20, decoder.build, self.conf)
        self.swa_Balancer      = SWA(20, balancer.build, self.conf)

        self.set_swa_model_weights()

    def set_swa_model_weights(self):
        self.swa_D_Mask.model        = self.model.D_Mask
        self.swa_D_Image1.model      = self.model.D_Image1
        self.swa_D_Image2.model      = self.model.D_Image2
        self.swa_Enc_Anatomy1.model  = self.model.Encoders_Anatomy[0]
        self.swa_Enc_Anatomy2.model  = self.model.Encoders_Anatomy[1]
        self.swa_Enc_Modality.model  = self.model.Enc_Modality
        self.swa_Anatomy_Fuser.model = self.model.Anatomy_Fuser
        self.swa_Segmentor.model     = self.model.Segmentor
        self.swa_Decoder.model       = self.model.Decoder
        self.swa_Balancer.model      = self.model.Balancer

    def init_train_data(self):
        self.gen_labelled = self._init_labelled_data_generator()
        self.gen_unlabelled      = self._init_unlabelled_data_generator()
        self.discriminator_masks = self._init_disciminator_mask_generator()
        self.discriminator_image = [self._init_discriminator_image_generator(mod)
                                    for mod in self.model.modalities]

        self.batches = int(np.ceil(self.data_len / self.conf.batch_size))

    def _init_labelled_data_generator(self):
        """
        Initialise a data generator (image, mask) for labelled data
        """
        if self.conf.l_mix == 0:
            return

        log.info('Initialising labelled datagen. Loading %s data' % self.conf.dataset_name)
        self.data = self.loader.load_all_modalities_concatenated(self.conf.split, 'training',self.conf.image_downsample)
        self.data.sample(int(np.round(self.conf.l_mix * self.data.num_volumes)), seed=self.conf.seed)

        if hasattr(self.conf, 'randomise') and self.conf.randomise:
            self.data.randomise_pairs(self.conf.n_pairs - 1, seed=self.conf.seed)
        elif self.conf.automatedpairing:
            self.data.expand_pairs(self.conf.n_pairs - 1, 0, neighborhood=self.conf.n_pairs)
            self.data.expand_pairs(self.conf.n_pairs - 1, 1, neighborhood=self.conf.n_pairs)

        log.info('labelled data size: ' + str(self.data.size()))
        self.data_len = self.data.size()

        return self.get_data_generator(train_images=[self.data.get_images_modi(i) for i in range(2)],
                                       train_labels=[self.data.get_masks_modi(i) for i in range(2)])

    def _init_unlabelled_data_generator(self):
        """
        Initialise a data generator (image) for unlabelled data
        """
        if self.conf.l_mix == 1:
            return

        self.ul_data = self._load_unlabelled_data('training', 'ul', None)
        self.conf.unlabelled_image_num = self.ul_data.size()
        if self.data is None or self.ul_data.size() > self.data.size():
            self.data_len = self.ul_data.size()

        return self.get_data_generator(train_images=[self.ul_data.get_images_modi(i) for i in range(2)],
                                       train_labels=[self.ul_data.get_masks_modi(0)])

    def _load_unlabelled_data(self, split_type, data_type, modality):
        """
        Create a Data object with unlabelled data. This will be used to train the unlabelled path of the
        generators and produce fake masks for training the discriminator
        :param split_type:  the split defining which volumes to load
        :param data_type:   can be one ['ul', 'all']. The second includes images that have masks.
        :return:            a data object
        """
        log.info('Initialising unlabelled datagen. Loading %s data of type %s' % (self.conf.dataset_name, data_type))
        if data_type == 'ul':
            log.info('Estimating number of unlabelled images from %s data' % self.conf.dataset_name)
            ul_data = self.loader.load_all_modalities_concatenated(self.conf.split, split_type,
                                                                   self.conf.image_downsample)
            self.conf.num_ul_volumes = ul_data.num_volumes

            if hasattr(self.conf, 'randomise') and self.conf.randomise:
                ul_data.randomise_pairs(length=self.conf.n_pairs - 1)
            elif self.conf.automatedpairing:
                ul_data.expand_pairs(self.conf.n_pairs - 1, 0, neighborhood=self.conf.n_pairs)
                ul_data.expand_pairs(self.conf.n_pairs - 1, 1, neighborhood=self.conf.n_pairs)

            if self.conf.l_mix > 0:
                num_lb_vols = int(np.round(self.conf.l_mix * ul_data.num_volumes))
                volumes = ul_data.get_sample_volumes(num_lb_vols, seed=self.conf.seed)
                ul_volumes = [v for v in ul_data.volumes() if v not in volumes] # ul volumes are the remaining from lbl
                ul_data.filter_volumes(ul_volumes)

            log.info('unlabelled data size: ' + str(ul_data.size()))
        elif data_type == 'all':
            ul_data = self.loader.load_all_data(self.conf.split, split_type, modality=modality,
                                                downsample=self.conf.image_downsample)
        else:
            raise Exception('Invalid data_type: %s' % str(data_type))

        return ul_data

    def _init_disciminator_mask_generator(self):
        """
        Init a generator for masks to use in the discriminator.
        """
        log.info('Initialising discriminator maskgen.')
        masks = self._load_discriminator_masks()
        return self.get_data_generator(train_images=None, train_labels=[masks])

    def _load_discriminator_masks(self):
        masks = []
        if self.data is not None:
            masks.append(np.concatenate([self.data.get_masks_modi(0), self.data.get_masks_modi(1)], axis=0))
        if self.ul_data is not None:
            masks.append(self.ul_data.get_masks_modi(0))

        if len(masks) == 0:
            masks = np.empty(shape=([0] + self.conf.input_shape[:-1] + [self.loader.num_masks]))
        else:
            masks = np.concatenate(masks, axis=0)

        im_shape = self.conf.input_shape[:2]
        assert masks.shape[1] == im_shape[0] and masks.shape[2] == im_shape[1], masks.shape

        return masks

    def _init_discriminator_image_generator(self, modality):
        """
        Init a generator for images to train a discriminator (for fake masks)
        """
        log.info('Initialising discriminator imagegen.')
        data = self._load_unlabelled_data('training', 'all', modality)
        return self.get_data_generator(train_images=[data.images], train_labels=None)

    def init_image_callback(self):
        log.info('Initialising a data generator to use for printing.')

        if self.data is None:
            data = self.loader.load_all_modalities_concatenated(self.conf.split, 'training', self.conf.image_downsample)
            if hasattr(self.conf, 'randomise') and self.conf.randomise:
                data.randomise_pairs(length=self.conf.n_pairs - 1)
        else:
            data = self.data

        gen = self.get_data_generator(train_images=[data.get_images_modi(i) for i in range(2)],
                                       train_labels=[data.get_masks_modi(i) for i in range(2)])
        self.img_callback = DAFNetImageCallback(self.conf, self.model, gen)

    def get_loss_names(self):
        return ['adv_M', 'adv_X1', 'adv_X2', 'rec_X', 'dis_M', 'dis_X1', 'dis_X2',
                'val_loss', 'val_loss_mod1', 'val_loss_mod2',
                'val_loss_mod2_mod1def', 'val_loss_mod1_mod2def', 'val_loss_mod2_fused', 'val_loss_mod1_fused',
                'val_weight_0', 'val_weight_1', 'val_weight_2',
                'supervised_Mask', 'KL', 'rec_Z']

    def get_swa_models(self):
        return [self.swa_D_Mask, self.swa_D_Image1, self.swa_D_Image2,
                self.swa_Enc_Anatomy1, self.swa_Enc_Anatomy2, self.swa_Enc_Modality,
                self.swa_Anatomy_Fuser, self.swa_Segmentor, self.swa_Decoder, self.swa_Balancer]

    def train(self):
        log.info('Training Model')

        self.init_train_data()

        self.init_image_callback()
        sl = SaveLoss(self.conf.folder)
        cl = CSVLogger(self.conf.folder + '/training.csv')
        cl.on_train_begin()

        es = EarlyStopping('val_loss_mod2', min_delta=0.01, patience=60)
        es.model = self.model.Segmentor
        es.on_train_begin()

        loss_names = self.get_loss_names()
        total_loss = {n: [] for n in loss_names}

        progress_bar = Progbar(target=self.batches * self.conf.batch_size)
        for self.epoch in range(self.conf.epochs):
            log.info('Epoch %d/%d' % (self.epoch, self.conf.epochs))

            epoch_loss = {n: [] for n in loss_names}
            epoch_loss_list = []

            for self.batch in range(self.batches):
                self.train_batch(epoch_loss)
                progress_bar.update((self.batch + 1) * self.conf.batch_size)

            self.set_swa_model_weights()
            for swa_m in self.get_swa_models():
                swa_m.on_epoch_end(self.epoch)

            self.validate(epoch_loss)

            for n in loss_names:
                epoch_loss_list.append((n, np.mean(epoch_loss[n])))
                total_loss[n].append(np.mean(epoch_loss[n]))
            log.info(str('Epoch %d/%d: ' + ', '.join([l + ' Loss = %.5f' for l in loss_names])) %
                     ((self.epoch, self.conf.epochs) + tuple(total_loss[l][-1] for l in loss_names)))
            logs = {l: total_loss[l][-1] for l in loss_names}

            cl.model = self.model.D_Mask
            cl.model.stop_training = False
            cl.on_epoch_end(self.epoch, logs)
            sl.on_epoch_end(self.epoch, logs)

            # print images
            self.img_callback.on_epoch_end(self.epoch)

            self.save_models()

            if self.stop_criterion(es, logs):
                log.info('Finished training from early stopping criterion')

                es.on_train_end(logs)
                cl.on_train_end(logs)
                for swa_m in self.get_swa_models():
                    swa_m.on_train_end()

                # Set final model parameters based on SWA
                self.model.D_Mask              = self.swa_D_Mask.model
                self.model.D_Image1            = self.swa_D_Image1.model
                self.model.D_Image2            = self.swa_D_Image2.model
                self.model.Encoders_Anatomy[0] = self.swa_Enc_Anatomy1.model
                self.model.Encoders_Anatomy[1] = self.swa_Enc_Anatomy2.model
                self.model.Enc_Modality        = self.swa_Enc_Modality.model
                self.model.Anatomy_Fuser       = self.swa_Anatomy_Fuser.model
                self.model.Segmentor           = self.swa_Segmentor.model
                self.model.Decoder             = self.swa_Decoder.model
                self.model.Balancer            = self.swa_Balancer.model

                self.save_models()
                break

    def save_models(self, postfix=''):
        log.debug('Saving trained models')
        model_folder = self.conf.folder + '/models/'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        self.swa_D_Mask.get_clone_model().save_weights(model_folder + 'D_Mask' + postfix)
        self.swa_D_Image1.get_clone_model().save_weights(model_folder + 'D_Image1' + postfix)
        self.swa_D_Image2.get_clone_model().save_weights(model_folder + 'D_Image2' + postfix)
        self.swa_Enc_Anatomy1.get_clone_model().save_weights(model_folder + 'Enc_Anatomy1' + postfix)
        self.swa_Enc_Anatomy2.get_clone_model().save_weights(model_folder + 'Enc_Anatomy2' + postfix)
        self.swa_Enc_Modality.get_clone_model().save_weights(model_folder + 'Enc_Modality' + postfix)
        self.swa_Anatomy_Fuser.get_clone_model().save_weights(model_folder + 'Anatomy_Fuser' + postfix)
        self.swa_Segmentor.get_clone_model().save_weights(model_folder + 'Segmentor' + postfix)
        self.swa_Decoder.get_clone_model().save_weights(model_folder + 'Decoder' + postfix)
        self.swa_Balancer.get_clone_model().save_weights(model_folder + 'Balancer' + postfix)

    def validate(self, epoch_loss):
        """
        Calculate losses on validation set
        :param epoch_loss: the dictionary to save the losses
        """
        valid_data = self.loader.load_all_modalities_concatenated(self.conf.split, 'validation',
                                                                  self.conf.image_downsample)
        if (hasattr(self.conf, 'randomise') and self.conf.randomise):
            valid_data.randomise_pairs(length=self.conf.n_pairs - 1)
        valid_data.crop(self.conf.input_shape[:2])

        images0 = valid_data.get_images_modi(0)
        images1 = valid_data.get_images_modi(1)
        masks0  = valid_data.get_masks_modi(0)
        masks1  = valid_data.get_masks_modi(1)

        s1 = self.swa_Enc_Anatomy1.get_clone_model().predict(images0)
        s2 = self.swa_Enc_Anatomy2.get_clone_model().predict(images1)
        z1, _ = self.swa_Enc_Modality.get_clone_model().predict([s1, images0])
        z2, _ = self.swa_Enc_Modality.get_clone_model().predict([s2, images1])

        swa_Anatomy_Fuser_model = self.swa_Anatomy_Fuser.get_clone_model()
        s1_deformed, s2_fused = swa_Anatomy_Fuser_model.predict([s1, s2])
        s2_deformed, s1_fused = swa_Anatomy_Fuser_model.predict([s2, s1])

        swa_Segmentor_model = self.swa_Segmentor.get_clone_model()

        m1_s1 = swa_Segmentor_model.predict(s1)
        m2_s2 = swa_Segmentor_model.predict(s2)
        m2_s1def = swa_Segmentor_model.predict(s1_deformed)
        m1_s2def = swa_Segmentor_model.predict(s2_deformed)
        m2_fused = swa_Segmentor_model.predict(s2_fused)
        m1_fused = swa_Segmentor_model.predict(s1_fused)

        assert m1_s1.shape[:-1] == images0.shape[:-1], str(images0) + ' ' + str(m1_s1.shape)
        assert m2_s2.shape[:-1] == images0.shape[:-1], str(images0.shape) + ' ' + str(m2_s2.shape)
        assert m2_s1def.shape[:-1] == images0.shape[:-1], str(images0.shape) + ' ' + str(m2_s1def.shape)

        dice_m1s1    = (1 - costs.dice(masks0, m1_s1, binarise=True))
        dice_m1s2def = (1 - costs.dice(masks0, m1_s2def, binarise=True))
        dice_m1fused = (1 - costs.dice(masks0, m1_fused, binarise=True))

        dice_m2s2    = (1 - costs.dice(masks1, m2_s2, binarise=True))
        dice_m2s1def = (1 - costs.dice(masks1, m2_s1def, binarise=True))
        dice_m2fused = (1 - costs.dice(masks1, m2_fused, binarise=True))
        epoch_loss['val_loss_mod2'].append(dice_m2s2)
        epoch_loss['val_loss_mod2_mod1def'].append(dice_m2s1def)
        epoch_loss['val_loss_mod2_fused'].append(dice_m2fused)
        epoch_loss['val_loss_mod1_mod2def'].append(dice_m1s2def)
        epoch_loss['val_loss_mod1_fused'].append(dice_m1fused)
        epoch_loss['val_loss_mod1'].append(dice_m1s1)
        epoch_loss['val_loss'].append(np.mean([dice_m1s1, dice_m2s2, dice_m2s1def, dice_m2fused]))

        if self.conf.automatedpairing:
            valid_data.expand_pairs(self.conf.n_pairs - 1, 0, neighborhood=self.conf.n_pairs)
            images0      = valid_data.get_images_modi(0)
            images0_list = [images0[..., i:i + 1] for i in range(images0.shape[-1])]
            images1 = valid_data.get_images_modi(1)
            s1_list = [self.model.Encoders_Anatomy[0].predict(x) for x in images0_list]
            s2      = self.model.Encoders_Anatomy[1].predict(images1)

            weights = self.model.Balancer.predict([s2] + s1_list)
            weights = [np.mean(weights[..., j]) for j in range(weights.shape[-1])]
            for j in range(len(weights)):
                epoch_loss['val_weight_%d' % j].append(weights[j])

    def train_batch(self, epoch_loss):
        if self.conf.automatedpairing:
            if self.conf.l_mix > 0:
                self.train_supervised_automated_pairing(epoch_loss)
                self.train_batch_mask_discriminator(epoch_loss)
                self.train_batch_image_discriminator(epoch_loss)
            if self.conf.l_mix < 1:
                self.train_batch_mask_discriminator(epoch_loss)
                self.train_batch_image_discriminator(epoch_loss)
                self.train_unsupervised_automated_pairing(epoch_loss)
        else:
            if self.conf.l_mix > 0:
                self.train_supervised_expert_pairing(epoch_loss)
                self.train_batch_mask_discriminator(epoch_loss)
                self.train_batch_image_discriminator(epoch_loss)
            if self.conf.l_mix < 1:
                self.train_batch_mask_discriminator(epoch_loss)
                self.train_batch_image_discriminator(epoch_loss)
                self.train_unsupervised_expert_pairing(epoch_loss)

    def train_supervised_expert_pairing(self, epoch_loss):
        """
        Train generator/segmentation networks.
        :param epoch_loss:  Dictionary of losses for the epoch
        """
        x1_pairs, x2_pairs, m1_pairs, m2_pairs = next(self.gen_labelled)
        m1, m2, x1, x1_list, x2, x2_list, z1, z2 = self.prepare_data_to_train(x1_pairs, x2_pairs, m1_pairs, m2_pairs)

        batch_size = x1.shape[0]  # maybe this differs from conf.batch_size at the last batch.
        dm_shape = (batch_size,) + self.model.D_Mask.output_shape[1:]
        dx_shape = (batch_size,) + self.model.D_Image1.output_shape[1:]
        ones_m = np.ones(shape=dm_shape)
        ones_x = np.ones(shape=dx_shape)
        zeros = np.zeros(shape=(batch_size,))

        h = self.model.supervised_trainer.fit([x1, x2, z1, z2],
                                              [m1, m2, m1, m2]           +  # supervised cost
                                              [ones_m for _ in range(4)] +  # mask adversarial
                                              [x1, x2, x1, x2]           +  # reconstruction cost
                                              [ones_x for _ in range(4)] +  # image adversarial
                                              [zeros for _ in range(2)]  +  # KL divergence
                                              [z1, z2], verbose=False)
        self.store_training_losses(h, epoch_loss)

    def train_unsupervised_expert_pairing(self, epoch_loss):
        """
        Train generator/segmentation networks.
        :param epoch_loss:  Dictionary of losses for the epoch
        """
        x1_pairs, x2_pairs, m1_pairs = next(self.gen_unlabelled)
        m1, _, x1, x1_list, x2, x2_list, z1, z2 = self.prepare_data_to_train(x1_pairs, x2_pairs, m1_pairs, None)

        batch_size = x1.shape[0]  # maybe this differs from conf.batch_size at the last batch.
        dm_shape = (batch_size,) + self.model.D_Mask.output_shape[1:]
        dx_shape = (batch_size,) + self.model.D_Image1.output_shape[1:]
        ones_m = np.ones(shape=dm_shape)
        ones_x = np.ones(shape=dx_shape)
        zeros  = np.zeros(shape=(batch_size,))
        h = self.model.unsupervised_trainer.fit([x1, x2, z1, z2],
                                              [m1, m1]                   +  # supervised cost
                                              [ones_m for _ in range(4)] +  # mask adversarial
                                              [x1, x2, x1, x2]           +  # reconstruction cost
                                              [ones_x for _ in range(4)] +  # image adversarial
                                              [zeros for _ in range(2)]  +  # KL divergence
                                              [z1, z2], verbose=False)
        self.store_training_losses(h, epoch_loss)

    def train_supervised_automated_pairing(self, epoch_loss):
        """
        Train generator/segmentation networks.
        :param epoch_loss:  Dictionary of losses for the epoch
        """
        x1_pairs, x2_pairs, m1_pairs, m2_pairs = next(self.gen_labelled)
        m1, m2, x1, x1_list, x2, x2_list, z1, z2 = self.prepare_data_to_train(x1_pairs, x2_pairs, m1_pairs, m2_pairs)

        batch_size = x1.shape[0]  # maybe this differs from conf.batch_size at the last batch.
        dm_shape = (batch_size,) + self.model.D_Mask.output_shape[1:]
        dx_shape = (batch_size,) + self.model.D_Image1.output_shape[1:]
        ones_m = np.ones(shape=dm_shape)
        ones_x = np.ones(shape=dx_shape)
        zeros = np.zeros(shape=(batch_size,))
        h = self.model.supervised_trainer.fit(x1_list + x2_list + [m1, m2, z1, z2],
                                              [m1, m2, zeros, zeros]     +  # supervised cost
                                              [ones_m for _ in range(4)] +  # mask adversarial
                                              [x1, x2, zeros, zeros]     +  # reconstruction cost
                                              [ones_x for _ in range(4)] +  # image adversarial
                                              [zeros for _ in range(2)]  +  # KL divergence
                                              [z1, z2], verbose=False)
        self.store_training_losses(h, epoch_loss)

    def train_unsupervised_automated_pairing(self, epoch_loss):
        """
        Train generator/segmentation networks.
        :param epoch_loss:  Dictionary of losses for the epoch
        """
        x1_pairs, x2_pairs, m1_pairs = next(self.gen_unlabelled)
        m1, _, x1, x1_list, x2, x2_list, z1, z2 = self.prepare_data_to_train(x1_pairs, x2_pairs, m1_pairs, None)

        batch_size = x1.shape[0]  # maybe this differs from conf.batch_size at the last batch.
        dm_shape = (batch_size,) + self.model.D_Mask.output_shape[1:]
        dx_shape = (batch_size,) + self.model.D_Image1.output_shape[1:]
        ones_m = np.ones(shape=dm_shape)
        ones_x = np.ones(shape=dx_shape)
        zeros  = np.zeros(shape=(batch_size,))
        h = self.model.unsupervised_trainer.fit(x1_list + x2_list + [m1, z1, z2],
                                              [m1, zeros]                + # supervised cost
                                              [ones_m for _ in range(4)] + # mask adversarial
                                              [x1, x2, zeros, zeros]     + # reconstruction cost
                                              [ones_x for _ in range(4)] + # image adversarial
                                              [zeros for _ in range(2)]  + # KL divergence
                                              [z1, z2], verbose=False)
        self.store_training_losses(h, epoch_loss)

    def prepare_data_to_train(self, x1_pairs, x2_pairs, m1_pairs, m2_pairs):
        if m2_pairs is not None:
            [x1_pairs, x2_pairs, m1_pairs, m2_pairs] = self.align_batches([x1_pairs, x2_pairs, m1_pairs, m2_pairs])
        else:
            [x1_pairs, x2_pairs, m1_pairs] = self.align_batches([x1_pairs, x2_pairs, m1_pairs])

        split_images = lambda x: [x[..., i:i + 1] for i in range(self.conf.n_pairs)]
        x1_list = split_images(x1_pairs)
        x2_list = split_images(x2_pairs)
        x1 = x1_list[0]
        x2 = x2_list[0]
        m1 = self.add_residual(m1_pairs[..., 0:self.loader.num_masks])
        m2 = self.add_residual(m2_pairs[..., 0:self.loader.num_masks]) if m2_pairs is not None else None

        batch_size = x1.shape[0]  # maybe this differs from conf.batch_size at the last batch.
        norm = NormalDistribution()
        z1 = norm.sample((batch_size, self.conf.num_z))
        z2 = norm.sample((batch_size, self.conf.num_z))
        return m1, m2, x1, x1_list, x2, x2_list, z1, z2

    def store_training_losses(self, h, epoch_loss):
        epoch_loss['supervised_Mask'].append(np.mean(h.history['Segmentor_loss']))
        epoch_loss['adv_M'].append(np.mean(h.history['D_Mask_loss']))
        epoch_loss['rec_X'].append(np.mean([h.history['Decoder_loss']]))
        epoch_loss['adv_X1'].append(np.mean(h.history['D_Image1_loss']))
        epoch_loss['adv_X2'].append(np.mean(h.history['D_Image2_loss']))
        epoch_loss['KL'].append(np.mean(h.history['Enc_Modality_loss']))
        epoch_loss['rec_Z'].append(np.mean(h.history['ZReconstruct_loss']))

    def train_batch_mask_discriminator(self, epoch_loss):
        """
        Jointly train a discriminator for images and masks.
        :param epoch_loss:  Dictionary of losses for the epoch
        """
        m1 = next(self.discriminator_masks)[..., 0:self.conf.num_masks]
        m2 = next(self.discriminator_masks)[..., 0:self.conf.num_masks]
        x1, x2 = self.align_batches([next(gen) for gen in self.discriminator_image])
        x1, x2, m1, m2 = self.align_batches([x1, x2, m1, m2])
        batch_size = x1.shape[0]  # maybe this differs from conf.batch_size at the last batch.

        m_shape = (batch_size,) + self.model.D_Mask.get_output_shape_at(0)[1:]

        fake_s1 = self.model.Encoders_Anatomy[0].predict(x1)
        fake_s2 = self.model.Encoders_Anatomy[1].predict(x2)

        # modality 1
        fake_m1 = self.model.Segmentor.predict(fake_s1)
        s2_def, _ = self.model.Anatomy_Fuser.predict([fake_s2, fake_s1])
        fake_m1_from_s2 = self.model.Segmentor.predict(s2_def)
        fake_m1_list = [m[..., 0:self.conf.num_masks] for m in [fake_m1, fake_m1_from_s2]]
        fake_m1 = np.concatenate(fake_m1_list, axis=0)

        # modality 2
        fake_m2 = self.model.Segmentor.predict(fake_s2)
        s1_def, _ = self.model.Anatomy_Fuser.predict([fake_s1, fake_s2])
        fake_m2_from_s1 = self.model.Segmentor.predict(s1_def)
        fake_m2_list = [m[..., 0:self.conf.num_masks] for m in [fake_m2, fake_m2_from_s1]]
        fake_m2 = np.concatenate(fake_m2_list, axis=0)

        real_m = np.concatenate([m1, m2], axis=0)
        real_m = utils.data_utils.sample(real_m, batch_size)

        fake_m = np.concatenate([fake_m1, fake_m2], axis=0)
        fake_m = utils.data_utils.sample(fake_m, batch_size)

        h = self.model.D_Mask_trainer.fit([real_m, fake_m], [np.ones(m_shape), np.zeros(m_shape)], epochs=1, verbose=0)
        epoch_loss['dis_M'].append(np.mean(h.history['loss']))

    def train_batch_image_discriminator(self, epoch_loss):
        """
        Train an image discriminator.
        :param epoch_loss:  Dictionary of losses for the epoch
        """
        x_list = self.align_batches([next(gen) for gen in self.discriminator_image])
        batch_size = x_list[0].shape[0]  # maybe this differs from conf.batch_size at the last batch.

        fake_s_list = [self.model.Encoders_Anatomy[i].predict(x_list[i]) for i in range(len(self.model.modalities))]

        x1, x2 = x_list
        s1, s2 = fake_s_list

        s1_def = self.model.Anatomy_Fuser.predict(fake_s_list)[0]
        s2_def = self.model.Anatomy_Fuser.predict(list(reversed(fake_s_list)))[0]
        z1, _  = self.model.Enc_Modality.predict([s1, x1])
        z2, _  = self.model.Enc_Modality.predict([s2, x2])

        y1a = self.model.Decoder.predict([s1, z1])
        y1b = self.model.Decoder.predict([s2_def, z1])
        y1c = self.model.Decoder.predict([s1_def, z1])
        y2a = self.model.Decoder.predict([s2, z2])
        y2b = self.model.Decoder.predict([s1_def, z2])
        y2c = self.model.Decoder.predict([s2_def, z2])

        y1 = np.concatenate([y1a, y1b, y1c], axis=0)
        y1 = utils.data_utils.sample(y1, batch_size)
        y2 = np.concatenate([y2a, y2b, y2c], axis=0)
        y2 = utils.data_utils.sample(y2, batch_size)

        # Train Discriminator
        x_shape = (batch_size,) + self.model.D_Image1.output_shape[1:]
        h = self.model.D_Image1_trainer.fit([x1, y1], [np.ones(x_shape), np.zeros(x_shape)], epochs=1, verbose=0)
        epoch_loss['dis_X1'].append(np.mean(h.history['loss']))

        h = self.model.D_Image2_trainer.fit([x2, y2], [np.ones(x_shape), np.zeros(x_shape)], epochs=1, verbose=0)
        epoch_loss['dis_X2'].append(np.mean(h.history['loss']))

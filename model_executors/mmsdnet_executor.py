import logging

import numpy as np
from comet_ml import Experiment
from keras.callbacks import CSVLogger, EarlyStopping
from keras.utils import Progbar

import costs
from callbacks.dafnet_image_callback import DAFNetImageCallback
from callbacks.loss_callback import SaveLoss
from model_executors.base_executor import Executor
from utils.distributions import NormalDistribution

log = logging.getLogger('mmsdnet_executor')


class MMSDNetExecutor(Executor):
    """
    Train a DAFNet or MMSDNet model using parameters stored in the configuration.
    """
    def __init__(self, conf, model):
        super(MMSDNetExecutor, self).__init__(conf, model)
        self.exp_clb = None
        self.loader.modalities = self.conf.modality

        self.gen_labelled = None  # iterator for labelled data (supervised learning)
        self.gen_unlabelled      = None  # iterator for unlabelled data (unsupervised learning)
        self.discriminator_masks = None  # iterator for real masks to train discriminators
        self.discriminator_image = None  # iterator for images to train discriminators
        self.img_callback = None         # callback to save images
        self.data = None                 # labelled data container of type MultimodalPairedData
        self.ul_data = None              # unlabelled data container of type MultimodalPairedData

        self.gen_unlabelled_lge  = None
        self.gen_unlabelled_cine = None

        self.M_pool = []  # Pool of masks

        self.img_callback = None
        self.conf.batches_lge = 0

    def init_train_data(self):
        self.gen_labelled = self._init_labelled_data_generator()
        self.gen_unlabelled    = self._init_unlabelled_data_generator()
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
        self.data = self.loader.load_all_modalities_concatenated(self.conf.split, 'training', self.conf.image_downsample)
        self.data.sample(int(np.round(self.conf.l_mix * self.data.num_volumes)), seed=self.conf.seed)
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
        '''
        Create a Data object with unlabelled data. This will be used to train the unlabelled path of the
        generators and produce fake masks for training the discriminator
        :param split_type:  the split defining which volumes to load
        :param data_type:   can be one ['ul', 'all']. The second includes images that have masks.
        :return:            a data object
        '''
        log.info('Initialising unlabelled datagen. Loading %s data of type %s' % (self.conf.dataset_name, data_type))
        if data_type == 'ul':
            log.info('Estimating number of unlabelled images from %s data' % self.conf.dataset_name)
            ul_data = self.loader.load_all_modalities_concatenated(self.conf.split, split_type,
                                                                   self.conf.image_downsample)
            self.conf.num_ul_volumes = ul_data.num_volumes

            if self.conf.l_mix > 0:
                num_lb_vols = int(np.round(self.conf.l_mix * ul_data.num_volumes))
                volumes = ul_data.get_sample_volumes(num_lb_vols, seed=self.conf.seed)
                ul_volumes = [v for v in ul_data.volumes() if v not in volumes]  # ul volumes are the remaining from lbl
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
        else:
            data = self.data

        gen = self.get_data_generator(train_images=[data.get_images_modi(i) for i in range(2)],
                                      train_labels=[data.get_masks_modi(i) for i in range(2)])
        self.img_callback = DAFNetImageCallback(self.conf, self.model, gen)

    def get_loss_names(self):
        return ['adv_M', 'rec_X', 'dis_M', 'val_loss', 'val_loss_mod1', 'val_loss_mod2',
                'val_loss_mod2_s1def', 'val_loss_mod2_fused', 'supervised_Mask', 'loss', 'KL', 'rec_Z']

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

            self.validate(epoch_loss)

            for n in loss_names:
                epoch_loss_list.append((n, np.mean(epoch_loss[n])))
                total_loss[n].append(np.mean(epoch_loss[n]))
            log.info(str('Epoch %d/%d: ' + ', '.join([l + ' Loss = %.3f' for l in loss_names])) %
                     ((self.epoch, self.conf.epochs) + tuple(total_loss[l][-1] for l in loss_names)))
            logs = {l: total_loss[l][-1] for l in loss_names}

            cl.model = self.model.D_Mask
            cl.model.stop_training = False
            cl.on_epoch_end(self.epoch, logs)
            sl.on_epoch_end(self.epoch, logs)

            # Plot some example images
            self.img_callback.on_epoch_end(self.epoch)

            self.model.save_models()

            if self.stop_criterion(es, logs):
                log.info('Finished training from early stopping criterion')
                break

    def validate(self, epoch_loss):
        # Report validation error
        valid_data = self.loader.load_all_modalities_concatenated(self.conf.split, 'validation', self.conf.image_downsample)
        valid_data.crop(self.conf.input_shape[:2])

        images0 = valid_data.get_images_modi(0)
        images1 = valid_data.get_images_modi(1)
        real_mask0 = valid_data.get_masks_modi(0)
        real_mask1 = valid_data.get_masks_modi(1)

        s1 = self.model.Encoders_Anatomy[0].predict(images0)
        s2 = self.model.Encoders_Anatomy[1].predict(images1)
        s1_deformed, s_fused = self.model.Anatomy_Fuser.predict([s1, s2])
        mask1 = self.model.Segmentor.predict(s1)
        mask2 = self.model.Segmentor.predict(s2)
        mask3 = self.model.Segmentor.predict(s1_deformed)
        mask4 = self.model.Segmentor.predict(s_fused)

        l_mod1       = (1 - costs.dice(real_mask0, mask1, binarise=True))
        l_mod2       = (1 - costs.dice(real_mask1, mask2, binarise=True))
        l_mod2_s1def = (1 - costs.dice(real_mask1, mask3, binarise=True))
        l_mod2_fused = (1 - costs.dice(real_mask1, mask4, binarise=True))
        epoch_loss['val_loss_mod2'].append(l_mod2)
        epoch_loss['val_loss_mod2_s1def'].append(l_mod2_s1def)
        epoch_loss['val_loss_mod2_fused'].append(l_mod2_fused)
        epoch_loss['val_loss_mod1'].append(l_mod1)
        epoch_loss['val_loss'].append(np.mean([l_mod1, l_mod2, l_mod2_s1def, l_mod2_fused]))

    def train_batch(self, epoch_loss):
        self.train_batch_generators(epoch_loss)
        self.train_batch_mask_discriminator(epoch_loss)

    def train_batch_generators(self, epoch_loss):
        """
        Train generator/segmentation networks.
        :param epoch_loss:  Dictionary of losses for the epoch
        """
        num_mod = len(self.model.modalities)

        if self.conf.l_mix > 0:
            x1, x2, m1, m2 = next(self.gen_labelled)
            [x1, x2, m1, m2] = self.align_batches([x1, x2, m1, m2])
            batch_size = x1.shape[0]  # maybe this differs from conf.batch_size at the last batch.
            dm_shape = (batch_size,) + self.model.D_Mask.output_shape[1:]
            ones_m = np.ones(shape=dm_shape)

            # Train labelled path (supervised_model)
            all_outputs = [m1, m2, m2, m2, m1, m1] + \
                          [ones_m for _ in range(num_mod * 3)] + \
                          [x1, x2, x2, x2, x1, x1] + \
                          [np.zeros(batch_size) for _ in range(num_mod * 3)]
            h = self.model.supervised_trainer.fit([x1, x2], all_outputs, epochs=1, verbose=0)
            epoch_loss['supervised_Mask'].append(np.mean(h.history['Segmentor_loss']))
            epoch_loss['adv_M'].append(np.mean(h.history['D_Mask_loss']))
            epoch_loss['rec_X'].append(np.mean(h.history['Decoder_loss']))
            epoch_loss['KL'].append(np.mean(h.history['Enc_Modality_loss']))

            # Train Z Regressor
            norm = NormalDistribution()
            s_list = [self.model.Encoders_Anatomy[i].predict(x) for i, x in enumerate([x1, x2])]
            s1_def, s1_fused = self.model.Anatomy_Fuser.predict(s_list)
            s2_def, s2_fused = self.model.Anatomy_Fuser.predict(list(reversed(s_list)))
            s_list += [s1_def, s1_fused]
            s_list += [s2_def, s2_fused]
            z_list = [norm.sample((batch_size, self.conf.num_z)) for _ in range(num_mod * 3)]
            h = self.model.Z_Regressor.fit(s_list + z_list, z_list, epochs=1, verbose=0)
            epoch_loss['rec_Z'].append(np.mean(h.history['loss']))

        # Train unlabelled path
        if self.conf.l_mix < 1:
            x1, x2, m1 = next(self.gen_unlabelled)
            [x1, x2, m1] = self.align_batches([x1, x2, m1])
            batch_size = x1.shape[0]  # maybe this differs from conf.batch_size at the last batch.
            dm_shape = (batch_size,) + self.model.D_Mask.output_shape[1:]
            ones_m = np.ones(shape=dm_shape)

            # Train unlabelled path (G_model)
            all_outputs = [m1, m1, m1] + \
                          [ones_m for _ in range(num_mod * 3)] + \
                          [x1, x2, x2, x2, x1, x1] + \
                          [np.zeros(batch_size) for _ in range(num_mod * 3)]
            h = self.model.unsupervised_trainer.fit([x1, x2], all_outputs, epochs=1, verbose=0)
            epoch_loss['supervised_Mask'].append(np.mean(h.history['Segmentor_loss']))
            epoch_loss['adv_M'].append(np.mean(h.history['D_Mask_loss']))
            epoch_loss['rec_X'].append(np.mean(h.history['Decoder_loss']))
            epoch_loss['KL'].append(np.mean(h.history['Enc_Modality_loss']))

            # Train Z Regressor
            norm = NormalDistribution()
            s_list = [self.model.Encoders_Anatomy[i].predict(x) for i, x in enumerate([x1, x2])]
            s1_def, s1_fused = self.model.Anatomy_Fuser.predict(s_list)
            s2_def, s2_fused = self.model.Anatomy_Fuser.predict(list(reversed(s_list)))
            s_list += [s1_def, s1_fused]
            s_list += [s2_def, s2_fused]
            z_list = [norm.sample((batch_size, self.conf.num_z)) for _ in range(num_mod * 3)]
            h = self.model.Z_Regressor.fit(s_list + z_list, z_list, epochs=1, verbose=0)
            epoch_loss['rec_Z'].append(np.mean(h.history['loss']))

    def train_batch_mask_discriminator(self, epoch_loss):
        """
        Jointly train a discriminator for images and masks.
        :param epoch_loss:  Dictionary of losses for the epoch
        """
        m = next(self.discriminator_masks)
        m = m[..., 0:self.conf.num_masks]
        x_list = self.align_batches([next(gen) for gen in self.discriminator_image])
        x_0, m = self.align_batches([x_list[0], m])
        x_list = self.align_batches([x_0] + x_list[1:])
        batch_size = m.shape[0]  # maybe this differs from conf.batch_size at the last batch.

        num_mod = len(self.model.modalities)
        fake_s_list = [self.model.Encoders_Anatomy[i].predict(x_list[i]) for i in range(num_mod)]
        fake_m_list = [self.model.Segmentor.predict(fake_s_list[i]) for i in range(num_mod)]
        s1_def, s1_fused = self.model.Anatomy_Fuser.predict(fake_s_list)
        fake_m_list += [self.model.Segmentor.predict(s) for s in [s1_def, s1_fused]]
        fake_m = np.concatenate(fake_m_list, axis=0)[..., 0:self.conf.num_masks]

        # Pool of fake images
        self.M_pool, fake_m = self.get_fake(fake_m, self.M_pool, sample_size=batch_size)

        # Train Discriminator
        m_shape = (batch_size,) + self.model.D_Mask.get_output_shape_at(0)[1:]
        h = self.model.D_Mask_trainer.fit([m, fake_m], [np.ones(m_shape), np.zeros(m_shape)], epochs=1, verbose=0)
        epoch_loss['dis_M'].append(np.mean(h.history['D_Mask_loss']))

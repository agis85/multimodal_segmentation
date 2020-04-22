import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from keras import Model

import utils
import utils.image_utils
import utils.data_utils
from callbacks.image_callback import BaseSaveImage, get_s0chn
from utils.distributions import NormalDistribution
from utils.sdnet_utils import get_net

log = logging.getLogger('callback')


class DAFNetImageCallback(BaseSaveImage):
    """
    Image callback for saving images during DAFNet training.
    Images are saved in a subfolder with name training_images, created inside the experiment folder.
    """
    def __init__(self, conf, model, data_gen_lb):
        """
        :param conf:        configuration object
        :param model:       a DAFNet model
        :param data_gen_lb: a python iterator of images+masks
        """
        self.conf = conf
        super(DAFNetImageCallback, self).__init__(conf.folder, model)
        self._make_dirs(self.folder)
        self.data_gen_lb = data_gen_lb
        self.init_models()

    def _make_dirs(self, folder):
        self.lr_folder = folder + '/images_lr'
        if not os.path.exists(self.lr_folder):
            os.makedirs(self.lr_folder)

        self.segm_folder = folder + '/images_segm'
        if not os.path.exists(self.segm_folder):
            os.makedirs(self.segm_folder)

        self.rec_folder = folder + '/images_rec'
        if not os.path.exists(self.rec_folder):
            os.makedirs(self.rec_folder)

        self.discr_folder = folder + '/images_discr'
        if not os.path.exists(self.discr_folder):
            os.makedirs(self.discr_folder)

    def init_models(self):
        self.encoders_anatomy = self.model.Encoders_Anatomy
        self.reconstructor = self.model.Decoder
        self.segmentor = self.model.Segmentor
        self.discr_mask = self.model.D_Mask
        self.enc_modality = self.model.Enc_Modality
        self.fuser = self.model.Anatomy_Fuser
        self.discr_mask = self.model.D_Mask

        mean = get_net(self.enc_modality, 'z_mean')
        var = get_net(self.enc_modality, 'z_log_var')
        self.z_mean = Model(self.enc_modality.inputs, mean.output)
        self.z_var = Model(self.enc_modality.inputs, var.output)

    def on_epoch_end(self, epoch=None, logs=None):
        '''
        Plot training images from the real_pool. For SDNet the real_pools will contain images paired with masks,
        and also unlabelled images.
        :param epoch:       current training epoch
        :param logs:
        '''
        x_mod1, x_mod2, m_mod1, m_mod2 = next(self.data_gen_lb)
        image_list = [x_mod1[..., 0:1], x_mod2[..., 0:1]]
        masks_list = [m_mod1[..., 0:self.conf.num_masks], m_mod2[..., 0:self.conf.num_masks]]

        # we usually plot 4 image-rows. If we have less, it means we've reached the end of the data, so iterate from
        # the beginning
        while len(image_list[0]) < 4:
            x_mod1, x_mod2 = image_list
            m_mod1, m_mod2 = masks_list

            x_mod1_2, x_mod2_2, m_mod1_2, m_mod2_2 = next(self.data_gen_lb)
            image_list = [np.concatenate([x_mod1[..., 0:1], x_mod1_2[..., 0:1]], axis=0),
                          np.concatenate([x_mod2[..., 0:1], x_mod2_2[..., 0:1]], axis=0)]
            masks_list = [np.concatenate([m_mod1[..., 0:self.conf.num_masks], m_mod1_2[..., 0:self.conf.num_masks]], axis=0),
                          np.concatenate([m_mod2[..., 0:self.conf.num_masks], m_mod2_2[..., 0:self.conf.num_masks]], axis=0)]

        self.plot_latent_representation(image_list, epoch)
        self.plot_segmentations(image_list, masks_list, epoch)
        self.plot_reconstructions(image_list, epoch)
        self.plot_discriminator_outputs(image_list, masks_list, epoch)

    def plot_latent_representation(self, image_list, epoch):
        """
        Plot a 4-row image, where the first column shows the input image and the following columns
        each of the 8 channels of the spatial latent representation.
        :param image_list:   a list of 4-dim arrays of images, one for each modality
        :param epoch     :   the epoch number
        """

        x_list, s_list = [], []
        for mod_i in range(len(image_list)):
            images = image_list[mod_i]

            x = utils.data_utils.sample(images, nb_samples=4, seed=self.conf.seed)
            x_list.append(x)

            # plot S
            s = self.encoders_anatomy[mod_i].predict(x)
            s_list.append(s)

            rows = [np.concatenate([x[i, :, :, 0]] + [s[i, :, :, s_chn] for s_chn in range(s.shape[-1])], axis=1)
                    for i in range(x.shape[0])]
            im_plot = np.concatenate(rows, axis=0)
            scipy.misc.imsave(self.lr_folder + '/mod_%d_s_lr_epoch_%d.png' % (mod_i, epoch), im_plot)

            # plot Z
            enc_modality_inputs = [self.encoders_anatomy[mod_i].predict(images), images]
            z, _ = self.enc_modality.predict(enc_modality_inputs)

            means = self.z_mean.predict(enc_modality_inputs)
            variances  = self.z_var.predict(enc_modality_inputs)
            means = np.var(means, axis=0)
            variances = np.mean(np.exp(variances), axis=0)
            with open(self.lr_folder + '/z_means.csv', 'a+') as f:
                f.writelines(', '.join([str(means[i]) for i in range(means.shape[0])]) + '\n')
            with open(self.lr_folder + '/z_vars.csv', 'a+') as f:
                f.writelines(', '.join([str(variances[i]) for i in range(variances.shape[0])]) + '\n')

        # plot deformed anatomies
        new_anatomies = self.fuser.predict(s_list)

        s1_def = new_anatomies[0]
        rows = [np.concatenate([x_list[0][i, :, :, 0], x_list[1][i, :, :, 0]] +
                               [s1_def[i, :, :, s_chn] for s_chn in range(s1_def.shape[-1])], axis=1)
                for i in range(x_list[0].shape[0])]
        im_plot = np.concatenate(rows, axis=0)
        scipy.misc.imsave(self.lr_folder + '/s1def_lr_epoch_%d.png' % (epoch), im_plot)

    def plot_segmentations(self, image_list, mask_list, epoch):
        '''
        Plot an image for every sample, where every row contains a channel of the spatial LR and a channel of the
        predicted mask.
        :param image_list:   a list of 4-dim arrays of images, one for each modality
        :param masks_list:   a list of 4-dim arrays of masks, one for each modality
        :param epoch:       the epoch number
        '''

        x_list, s_list, m_list2 = [], [], []
        for mod_i in range(len(image_list)):
            images = image_list[mod_i]
            masks  = mask_list[mod_i]

            x = utils.data_utils.sample(images, 4, seed=self.conf.seed)
            m = utils.data_utils.sample(masks, 4, seed=self.conf.seed)

            x_list.append(x)
            m_list2.append(m)

            assert x.shape[:-1] == m.shape[:-1], 'Incompatible shapes: %s vs %s' % (str(x.shape), str(m.shape))

            s = self.encoders_anatomy[mod_i].predict(x)
            y = self.segmentor.predict(s)

            s_list.append(s)

            rows = []
            for i in range(x.shape[0]):
                y_list = [y[i, :, :, chn] for chn in range(y.shape[-1])]
                m_list = [m[i, :, :, chn] for chn in range(m.shape[-1])]
                if m.shape[-1] < y.shape[-1]:
                    m_list += [np.zeros(shape=(m.shape[1], m.shape[2]))] * (y.shape[-1] - m.shape[-1])
                assert len(y_list) == len(m_list), 'Incompatible sizes: %d vs %d' % (len(y_list), len(m_list))
                rows += [np.concatenate([x[i, :, :, 0]] + y_list + m_list, axis=1)]
            im_plot = np.concatenate(rows, axis=0)
            scipy.misc.imsave(self.segm_folder + '/mod_%d_segmentations_epoch_%d.png' % (mod_i, epoch), im_plot)

        new_anatomies = self.fuser.predict(s_list)
        pred_masks    = [self.segmentor.predict(s) for s in new_anatomies]

        rows = []
        for i in range(x_list[0].shape[0]):
            for y in pred_masks:
                y_list = [y[i, :, :, chn] for chn in range(self.conf.num_masks)]
                m_list = [m_list2[1][i, :, :, chn] for chn in range(self.conf.num_masks)]
                assert len(y_list) == len(m_list), 'Incompatible sizes: %d vs %d' % (len(y_list), len(m_list))
                rows += [np.concatenate([x_list[0][i, :, :, 0], x_list[1][i, :, :, 0]] + y_list + m_list, axis=1)]
        im_plot = np.concatenate(rows, axis=0)
        scipy.misc.imsave(self.segm_folder + '/fused_segmentations_epoch_%d.png' % (epoch), im_plot)

    def plot_discriminator_outputs(self, image_list, mask_list, epoch):
        '''
        Plot a histogram of predicted values by the discriminator
        :param image_list:   a list of 4-dim arrays of images, one for each modality
        :param masks_list:   a list of 4-dim arrays of masks, one for each modality
        :param epoch:       the epoch number
        '''

        s_list = [enc.predict(x) for enc, x in zip(self.encoders_anatomy, image_list)]


        s_list += self.fuser.predict(s_list)
        # s2_def, fused_s2 = self.fuser.predict(reversed(s_list))
        # s_list += [s1_def, fused_s1]

        s = np.concatenate(s_list, axis=0)
        m = np.concatenate(mask_list, axis=0)
        pred_m = self.segmentor.predict(s)

        m      = m[..., 0:self.discr_mask.input_shape[-1]]
        pred_m = pred_m[..., 0:self.discr_mask.input_shape[-1]]

        m = utils.data_utils.sample(m, nb_samples=4)
        pred_m = utils.data_utils.sample(pred_m, nb_samples=4)

        plt.figure()
        for i in range(4):
            plt.subplot(4, 2, 2 * i + 1)
            m_allchn = np.concatenate([m[i, :, :, chn] for chn in range(m.shape[-1])], axis=1)
            plt.imshow(m_allchn, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_mask.predict(m[i:i + 1]).reshape(1, -1).mean(axis=1))

            plt.subplot(4, 2, 2 * i + 2)
            pred_m_allchn_img = np.concatenate([pred_m[i, :, :, chn] for chn in range(pred_m.shape[-1])], axis=1)
            plt.imshow(pred_m_allchn_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_mask.predict(pred_m).reshape(1, -1).mean(axis=1))
        plt.tight_layout()
        plt.savefig(self.discr_folder + '/discriminator_mask_epoch_%d.png' % epoch)
        plt.close()

    def plot_reconstructions(self, image_list, epoch):
        """
        Plot two images showing the combination of the spatial and modality LR to generate an image. The first
        image uses the predicted S and Z and the second samples Z from a Gaussian.
        :param image_list:  a list of 2 4-dim arrays of images
        :param epoch:       the epoch number
        """
        x_list, s_list = [], []
        for mod_i in range(len(image_list)):
            images = image_list[mod_i]
            x = utils.data_utils.sample(images, nb_samples=4, seed=self.conf.seed)
            x_list.append(x)

            # S + Z -> Image
            s = self.encoders_anatomy[mod_i].predict(x)
            s_list.append(s)

            im_plot = self.get_rec_image(x, s)
            scipy.misc.imsave(self.rec_folder + '/mod_%d_rec_epoch_%d.png' % (mod_i, epoch), im_plot)

        new_anatomies = self.fuser.predict(s_list)
        s1_def = new_anatomies[0]

        im_plot = self.get_rec_image(x_list[1], s1_def)
        scipy.misc.imsave(self.rec_folder + '/s1def_rec_epoch_%d.png' % (epoch), im_plot)

    def get_rec_image(self, x, s):
        z, _ = self.enc_modality.predict([s, x])
        gaussian = NormalDistribution()

        y = self.reconstructor.predict([s, z])
        y_s0 = self.reconstructor.predict([s, np.zeros(z.shape)])
        all_bkg = np.concatenate([np.zeros(s.shape[:-1] + (s.shape[-1] - 1,)), np.ones(s.shape[:-1] + (1,))], axis=-1)
        y_0z = self.reconstructor.predict([all_bkg, z])
        y_00 = self.reconstructor.predict([all_bkg, np.zeros(z.shape)])
        z_random = gaussian.sample(z.shape)
        y_random = self.reconstructor.predict([s, z_random])
        rows = [np.concatenate([x[i, :, :, 0], y[i, :, :, 0], y_random[i, :, :, 0], y_s0[i, :, :, 0]] +
                               [self.reconstructor.predict([get_s0chn(k, s), z])[i, :, :, 0] for k in
                                range(s.shape[-1] - 1)] +
                               [y_0z[i, :, :, 0], y_00[i, :, :, 0]], axis=1) for i in range(x.shape[0])]
        header = utils.image_utils.makeTextHeaderImage(x.shape[2], ['X', 'rec(s,z)', 'rec(s,~z)', 'rec(s,0)'] +
                                                       ['rec(s0_%d, z)' % k for k in range(s.shape[-1] - 1)] + [
                                                           'rec(0, z)', 'rec(0,0)'])
        im_plot = np.concatenate([header] + rows, axis=0)
        im_plot = np.clip(im_plot, -1, 1)
        return im_plot
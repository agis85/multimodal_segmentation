
import logging
import os
import numpy as np
import scipy

import costs
from loaders import loader_factory

log = logging.getLogger('model_tester')


class ModelTester(object):

    def __init__(self, model, conf):
        self.model = model
        self.conf  = conf

    def run(self):
        for modi, mod in enumerate(self.model.modalities):
            log.info('Evaluating model on test data for %s' % mod)
            self.test_modality(mod, modi)

    def make_test_folder(self, modality, suffix=''):
        folder = os.path.join(self.conf.folder, 'test_results_%s_%s_%s' % (self.conf.test_dataset, modality, suffix))
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder

    def test_modality(self, modality, modality_index):
        """
        Evaluate model on a given modality
        :param modality: the modality to load
        """
        test_loader = loader_factory.init_loader(self.conf.test_dataset)
        test_loader.modalities = self.conf.modality
        test_data   = test_loader.load_all_modalities_concatenated(self.conf.split, 'test', self.conf.image_downsample)
        test_data.crop(self.conf.input_shape[:2])  # crop data to input shape

        for type in ['simple', 'def', 'max']:
            folder = self.make_test_folder(modality, suffix=type)
            self.test_modality_type(folder, modality_index, type, test_loader, test_data)

        test_data.randomise_pairs(length=2, seed=self.conf.seed)
        for type in ['simple', 'def', 'max']:
            folder = self.make_test_folder(modality, suffix=type + '_rand')
            self.test_modality_type(folder, modality_index, type, test_loader, test_data)

    def test_modality_type(self, folder, modality_index, type, test_loader, test_data):
        assert type in ['simple', 'def', 'max', 'maxnostn']

        samples   = os.path.join(folder, 'samples')
        if not os.path.exists(samples):
            os.makedirs(samples)

        synth = []
        im_dice = {}

        f = open(os.path.join(folder, 'results.csv'), 'w')
        f.writelines('Vol, Dice, ' + ', '.join(['Dice%d' % mi for mi in range(test_loader.num_masks)]) + '\n')
        for vol_i in test_data.volumes():
            vol_folder = os.path.join(samples, 'vol_%s' % str(vol_i))
            if not os.path.exists(vol_folder):
                os.makedirs(vol_folder)

            vol_image_mod1 = test_data.get_volume_images_modi(0, vol_i)
            vol_image_mod2 = test_data.get_volume_images_modi(1, vol_i)
            assert vol_image_mod1.shape[0] > 0

            vol_mask = test_data.get_volume_masks_modi(modality_index, vol_i)
            prd_mask = self.model.predict_mask(modality_index, type, [vol_image_mod1, vol_image_mod2])

            synth.append(prd_mask)
            im_dice[vol_i] = costs.dice(vol_mask, prd_mask, binarise=True)
            sep_dice = [costs.dice(vol_mask[..., mi:mi + 1], prd_mask[..., mi:mi + 1], binarise=True)
                        for mi in range(test_loader.num_masks)]

            s = '%s, %.3f, ' + ', '.join(['%.3f'] * test_loader.num_masks) + '\n'
            d = (str(vol_i), im_dice[vol_i]) + tuple(sep_dice)
            f.writelines(s % d)

            self.plot_images(samples, vol_i, modality_index, prd_mask, vol_mask, [vol_image_mod1, vol_image_mod2])

        print('%s - Dice score: %.3f' % (type, np.mean(list(im_dice.values()))))
        f.close()

    def plot_images(self, samples, vol_i, modality_index, prd_mask, vol_mask, image_list):
        vol_image_mod2 = image_list[modality_index]

        for i in range(vol_image_mod2.shape[0]):
            vol_folder = os.path.join(samples, 'vol_%s' % str(vol_i))
            if not os.path.exists(vol_folder):
                os.makedirs(vol_folder)

            row1 = [vol_image_mod2[i, :, :, 0]] + [prd_mask[i, :, :, j] for j in range(vol_mask.shape[-1])]
            row2 = [vol_image_mod2[i, :, :, 0]] + [vol_mask[i, :, :, j] for j in range(vol_mask.shape[-1])]

            row1 = np.concatenate(row1, axis=1)
            row2 = np.concatenate(row2, axis=1)
            im = np.concatenate([row1, row2], axis=0)

            scipy.misc.imsave(os.path.join(vol_folder, 'test_vol%s_im%d.png' % (str(vol_i), i)), im)

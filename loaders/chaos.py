import logging
import os
import sys

import numpy as np
from scipy.ndimage import imread
from skimage import transform

sys.path.append('loaders')
sys.path.append('.')
from loaders.MultimodalPairedData import MultimodalPairedData
from loaders.base_loader import Loader, data_conf
from loaders.data import Data
from loaders.dcm_contour_utils import DicomImage
from utils import data_utils
import nibabel as nib
log = logging.getLogger('chaos')


class ChaosLoader(Loader):
    # average resolution 1.61
    def __init__(self):
        self.volumes = [1,2,3,5,8,10,13,15,19,20,21,22,31,32,33,34,36,37,38,39]
        super(ChaosLoader, self).__init__(self.volumes)
        self.num_masks = 4 # liver, right kidney, left kidney, spleen
        self.input_shape = (192, 192, 1)
        self.data_folder = data_conf['chaos']
        self.num_volumes = len(self.volumes)
        self.log = logging.getLogger('chaos')
        self.modalities = ['t1', 't2']

    def splits(self):
        return [
            {'validation': [31, 36, 13],
             'test': [10, 22, 34],
             'training': [5, 3, 1, 15, 19, 2, 20, 37, 32, 38, 8, 39, 21, 33]
             },
            {
                'validation': [13, 3, 20],
                'test': [5, 15, 39],
                'training': [33, 8, 38, 34, 36, 31, 32, 37, 22, 2, 1, 10, 19, 21]
            },
            {
                'validation': [37, 13, 33],
                'test': [1, 19, 32],
                'training': [5, 20, 31, 2, 38, 3, 8, 15, 22, 10, 34, 39, 36, 21]
            }
        ]

    def load_all_data(self, split, split_type, modality, normalise=True, downsample=1):
        return self.load_labelled_data(split, split_type, modality, normalise, downsample)

    def load_unlabelled_data(self, split, split_type, modality, normalise=True, downsample=1):
        return self.load_labelled_data(split, split_type, modality, normalise, downsample)

    def load_labelled_data(self, split, split_type, modality, normalise=True, downsample=1, root_folder=None):
        data = self.load_all_modalities_concatenated(split, split_type, downsample)
        images_t1 = data.get_images_modi(0)
        images_t2 = data.get_images_modi(1)
        labels_t1 = data.get_masks_modi(0)
        labels_t2 = data.get_masks_modi(1)

        if modality == 'all':
            all_images = np.concatenate([images_t1, images_t2], axis=0)
            all_labels = np.concatenate([labels_t1, labels_t2], axis=0)
            all_index  = np.concatenate([data.index, data.index.copy()], axis=0)
        elif modality == 't1':
            all_images = images_t1
            all_labels = labels_t1
            all_index  = data.index
        elif modality == 't2':
            all_images = images_t2
            all_labels = labels_t2
            all_index  = data.index
        else:
            raise Exception('Unknown modality: %s' % modality)

        assert split_type in ['training', 'validation', 'test', 'all'], split_type
        assert all_images.max() - 1 < 0.01 and all_images.min() + 1 < 0.01, \
            'max: %.3f, min: %.3f' % (all_images.max(), all_images.min())

        self.log.debug('Loaded compressed data of shape: ' + str(all_images.shape) + ' ' + str(all_index.shape))

        if split_type == 'all':
            return Data(all_images, all_labels, all_index, 1)

        volumes = self.splits()[split][split_type]
        all_images = np.concatenate([all_images[all_index == v] for v in volumes])

        assert all_labels.max() == 1 and all_labels.min() == 0, \
            'max: %d - min: %d' % (all_labels.max(), all_labels.min())

        all_masks = np.concatenate([all_labels[all_index == v] for v in volumes])
        assert all_images.shape[0] == all_masks.shape[0]
        all_index = np.concatenate([all_index[all_index == v] for v in volumes])
        assert all_images.shape[0] == all_index.shape[0]

        self.log.debug(split_type + ' set: ' + str(all_images.shape))
        return Data(all_images, all_masks, all_index, 1)

    def load_all_modalities_concatenated(self, split, split_type, downsample=1):
        all_images_t1, all_labels_t1, all_images_t2, all_labels_t2, all_index = [], [], [], [], []
        volumes = self.get_volumes_for_split(split, split_type)
        for v in volumes:
            images_t1, labels_t1 = self._load_volume(v, 't1')
            images_t2, labels_t2 = self._load_volume(v, 't2')

            # for each CHAOS subject, create pairs of T1 and T2 slices that approximately correspond to the same
            # position in the 3D volume, i.e. contain the same anatomical parts
            if v == 1:
                images_t2 = images_t2[1:]
                labels_t2 = labels_t2[1:]

                images_t1 = images_t1[0:26]
                labels_t1 = labels_t1[0:26]
                images_t2 = images_t2[4:24]
                labels_t2 = labels_t2[4:24]

                images_t1 = np.concatenate([images_t1[0:5], images_t1[7:10], images_t1[13:17], images_t1[18:]], axis=0)
                labels_t1 = np.concatenate([labels_t1[0:5], labels_t1[7:10], labels_t1[13:17], labels_t1[18:]], axis=0)
            if v == 2:
                images_t1 = np.concatenate([images_t1[4:7], images_t1[8:23]], axis=0)
                labels_t1 = np.concatenate([labels_t1[4:7], labels_t1[8:23]], axis=0)
                images_t2 = images_t2[3:22]
                labels_t2 = labels_t2[3:22]

                images_t1 = np.concatenate([images_t1[0:11], images_t1[12:18]], axis=0)
                labels_t1 = np.concatenate([labels_t1[0:11], labels_t1[12:18]], axis=0)
                images_t2 = np.concatenate([images_t2[0:11], images_t2[12:18]], axis=0)
                labels_t2 = np.concatenate([labels_t2[0:11], labels_t2[12:18]], axis=0)
            if v == 3:
                images_t1 = np.concatenate([images_t1[11:14], images_t1[15:26]], axis=0)
                labels_t1 = np.concatenate([labels_t1[11:14], labels_t1[15:26]], axis=0)
                images_t2 = images_t2[9:23]
                labels_t2 = labels_t2[9:23]
            if v == 5:
                images_t1 = np.concatenate([images_t1[4:5], images_t1[8:24]], axis=0)
                labels_t1 = np.concatenate([labels_t1[4:5], labels_t1[8:24]], axis=0)
                images_t2 = images_t2[2:22]
                labels_t2 = labels_t2[2:22]

                images_t2 = np.concatenate([images_t2[0:6], images_t2[9:]], axis=0)
                labels_t2 = np.concatenate([labels_t2[0:6], labels_t2[9:]], axis=0)

                images_t1 = np.concatenate([images_t1[0:8], images_t1[9:]], axis=0)
                labels_t1 = np.concatenate([labels_t1[0:8], labels_t1[9:]], axis=0)
                images_t2 = np.concatenate([images_t2[0:8], images_t2[9:]], axis=0)
                labels_t2 = np.concatenate([labels_t2[0:8], labels_t2[9:]], axis=0)
            if v == 8:
                images_t1 = images_t1[2:-2]
                labels_t1 = labels_t1[2:-2]

                images_t1 = np.concatenate([images_t1[5:11], images_t1[12:27]], axis=0)
                labels_t1 = np.concatenate([labels_t1[5:11], labels_t1[12:27]], axis=0)
                images_t2 = images_t2[6:27]
                labels_t2 = labels_t2[6:27]
            if v == 10:
                images_t1 = images_t1[14:38]
                labels_t1 = labels_t1[14:38]
                images_t2 = images_t2[5:24]
                labels_t2 = labels_t2[5:24]

                images_t1 = np.concatenate([images_t1[0:8], images_t1[12:18], images_t1[19:]], axis=0)
                labels_t1 = np.concatenate([labels_t1[0:8], labels_t1[12:18], labels_t1[19:]], axis=0)
            if v == 13:
                images_t1 = images_t1[4:29]
                labels_t1 = labels_t1[4:29]
                images_t2 = images_t2[3:28]
                labels_t2 = labels_t2[3:28]
            if v == 15:
                images_t1 = images_t1[:22]
                labels_t1 = labels_t1[:22]
                images_t2 = images_t2[:22]
                labels_t2 = labels_t2[:22]
            if v == 19:
                images_t1 = images_t1[8:27]
                labels_t1 = labels_t1[8:27]
                images_t2 = images_t2[5:24]
                labels_t2 = labels_t2[5:24]
            if v == 20:
                images_t1 = images_t1[2:21]
                labels_t1 = labels_t1[2:21]
                images_t2 = images_t2[2:21]
                labels_t2 = labels_t2[2:21]
            if v == 21:
                images_t1 = images_t1[3:19]
                labels_t1 = labels_t1[3:19]
                images_t2 = images_t2[5:21]
                labels_t2 = labels_t2[5:21]
            if v == 22:
                images_t1 = images_t1[:-2]
                labels_t1 = labels_t1[:-2]

                images_t1 = np.concatenate([images_t1[8:17], images_t1[18:26]], axis=0)
                labels_t1 = np.concatenate([labels_t1[8:17], labels_t1[18:26]], axis=0)
                images_t2 = np.concatenate([images_t2[3:12], images_t2[15:23]], axis=0)
                labels_t2 = np.concatenate([labels_t2[3:12], labels_t2[15:23]], axis=0)
            if v == 31:
                images_t1 = images_t1[7:23]
                labels_t1 = labels_t1[7:23]
                images_t2 = np.concatenate([images_t2[5:12], images_t2[13:22]], axis=0)
                labels_t2 = np.concatenate([labels_t2[5:12], labels_t2[13:22]], axis=0)
            if v == 32:
                images_t1 = images_t1[5:32]
                labels_t1 = labels_t1[5:32]

                images_t2 = images_t2[3:30]
                labels_t2 = labels_t2[3:30]
            if v == 33:
                images_t1 = images_t1[7:-5]
                labels_t1 = labels_t1[7:-5]
                images_t2 = np.concatenate([images_t2[3:12], images_t2[15:-2]], axis=0)
                labels_t2 = np.concatenate([labels_t2[3:12], labels_t2[15:-2]], axis=0)
            if v == 34:
                images_t1 = np.concatenate([images_t1[1:2], images_t1[3:4], images_t1[5:6], images_t1[7:27]], axis=0)
                labels_t1 = np.concatenate([labels_t1[1:2], labels_t1[3:4], labels_t1[5:6], labels_t1[7:27]], axis=0)
                images_t1 = np.concatenate([images_t1[0:14], images_t1[15:16], images_t1[17:18], images_t1[19:22], images_t1[23:24]], axis=0)
                labels_t1 = np.concatenate([labels_t1[0:14], labels_t1[15:16], labels_t1[17:18], labels_t1[19:22], labels_t1[23:24]], axis=0)
                images_t2 = images_t2[2:21]
                labels_t2 = labels_t2[2:21]
            if v == 36:
                images_t1 = images_t1[8:25]
                labels_t1 = labels_t1[8:25]
                images_t2 = np.concatenate([images_t2[4:6], images_t2[7:22]], axis=0)
                labels_t2 = np.concatenate([labels_t2[4:6], labels_t2[7:22]], axis=0)
            if v == 37:
                images_t1 = np.concatenate([images_t1[9:23], images_t1[24:-1]], axis=0)
                labels_t1 = np.concatenate([labels_t1[9:23], labels_t1[24:-1]], axis=0)
                images_t2 = np.concatenate([images_t2[4:6], images_t2[7:21], images_t2[22:-7]], axis=0)
                labels_t2 = np.concatenate([labels_t2[4:6], labels_t2[7:21], labels_t2[22:-7]], axis=0)
            if v == 38:
                images_t1 = images_t1[9:24]
                labels_t1 = labels_t1[9:24]
                images_t2 = images_t2[9:24]
                labels_t2 = labels_t2[9:24]
            if v == 39:
                images_t1 = images_t1[3:22]
                labels_t1 = labels_t1[3:22]
                images_t2 = images_t2[3:22]
                labels_t2 = labels_t2[3:22]

            images_t1 = np.concatenate([data_utils.rescale(images_t1[i:i + 1], -1, 1) for i in range(images_t1.shape[0])])
            images_t2 = np.concatenate([data_utils.rescale(images_t2[i:i + 1], -1, 1) for i in range(images_t2.shape[0])])

            assert images_t1.max() == 1 and images_t1.min() == -1, '%.3f to %.3f' % (images_t1.max(), images_t1.min())
            assert images_t2.max() == 1 and images_t2.min() == -1, '%.3f to %.3f' % (images_t2.max(), images_t2.min())

            all_images_t1.append(images_t1)
            all_labels_t1.append(labels_t1)
            all_images_t2.append(images_t2)
            all_labels_t2.append(labels_t2)

            all_index.append(np.array([v] * images_t1.shape[0]))

        all_images_t1, all_labels_t1 = data_utils.crop_same(all_images_t1, all_labels_t1, self.input_shape[:-1])
        all_images_t2, all_labels_t2 = data_utils.crop_same(all_images_t2, all_labels_t2, self.input_shape[:-1])

        all_images_t1 = np.concatenate(all_images_t1, axis=0)
        all_labels_t1 = np.concatenate(all_labels_t1, axis=0)
        all_images_t2 = np.concatenate(all_images_t2, axis=0)
        all_labels_t2 = np.concatenate(all_labels_t2, axis=0)

        if self.modalities == ['t1', 't2']:
            all_images = np.concatenate([all_images_t1, all_images_t2], axis=-1)
            all_labels = np.concatenate([all_labels_t1, all_labels_t2], axis=-1)
        elif self.modalities == ['t2', 't1']:
            all_images = np.concatenate([all_images_t2, all_images_t1], axis=-1)
            all_labels = np.concatenate([all_labels_t2, all_labels_t1], axis=-1)
        else:
            raise ValueError('invalid self.modalities', self.modalities)
        all_index = np.concatenate(all_index, axis=0)

        assert all_labels.max() == 1 and all_labels.min() == 0, '%.3f to %.3f' % (all_labels.max(), all_labels.min())
        return MultimodalPairedData(all_images, all_labels, all_index, downsample=downsample)

    def _load_volume(self, volume, modality):
        if modality == 't1':
            folder = self.data_folder + '/%d/T1DUAL' % volume
            image_folder = folder + '/DICOM_anon/OutPhase'
        elif modality == 't2':
            folder = self.data_folder + '/%d/T2SPIR' % volume
            image_folder = folder + '/DICOM_anon'
        else:
            raise Exception('Unknown modality')
        labels_folder = folder + '/Ground'

        image_files = list(os.listdir(image_folder))
        image_files.sort(key=lambda x: x.split('-')[-1], reverse=True)
        images_dcm = [DicomImage(image_folder + '/' + f) for f in image_files]
        images = np.concatenate([np.expand_dims(np.expand_dims(dcm.image, 0), -1) for dcm in images_dcm], axis=0)

        label_files = list(os.listdir(labels_folder))
        label_files.sort(key=lambda x: x.split('-')[-1], reverse=True)
        labels = [imread(labels_folder + '/' + f) for f in label_files]
        labels = np.concatenate([np.expand_dims(np.expand_dims(l, 0), -1) for l in labels], axis=0)

        res = images_dcm[0].resolution[0:2]
        images = np.concatenate([np.expand_dims(resample(images[i], res), axis=0) for i in range(images.shape[0])],
                                axis=0)
        labels = np.concatenate([np.expand_dims(resample(labels[i], res, binary=True), axis=0)
                                 for i in range(labels.shape[0])], axis=0)

        labels_l1 = labels.copy()
        labels_l1[labels != 63] = 0
        labels_l1[labels == 63] = 1

        labels_l2 = labels.copy()
        labels_l2[labels != 126] = 0
        labels_l2[labels == 126] = 1

        labels_l3 = labels.copy()
        labels_l3[labels != 189] = 0
        labels_l3[labels == 189] = 1

        labels_l4 = labels.copy()
        labels_l4[labels != 252] = 0
        labels_l4[labels == 252] = 1

        labels = np.concatenate([labels_l1, labels_l2, labels_l3, labels_l4], axis=-1)

        return images, labels


def resample(image, old_res, binary=False):
    """
    Resample all volumes to the same resolution
    :param image:   an image slice
    :param old_res: the original image resolution
    :param binary:  flag to denote a segmentation mask
    :return:        a resampled image
    """
    new_res = (1.89, 1.89)
    scale_vector = (old_res[0] / new_res[0], old_res[1] / new_res[1])
    order = 0 if binary else 1

    assert len(image.shape) == 3

    result = []
    for i in range(image.shape[-1]):
        im = image[..., i]
        rescaled = transform.rescale(im, scale_vector, order=order, preserve_range=True, mode='constant')
        result.append(np.expand_dims(rescaled, axis=-1))
    return np.concatenate(result, axis=-1)

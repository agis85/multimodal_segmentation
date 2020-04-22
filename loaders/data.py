import logging
import os

import numpy as np
from skimage.measure import block_reduce

import utils.data_utils
import utils.image_utils

log = logging.getLogger('data')


class Data(object):
    def __init__(self, images, masks, index, downsample=1):
        """
        Data constructor.
        :param images:      a 4-D numpy array of images. Expected shape: (N, H, W, 1)
        :param masks:       a 4-D numpy array of segmentation masks. Expected shape: (N, H, W, L)
        :param index:       a 1-D numpy array indicating the volume each image/mask belongs to. Used for data selection.
        :param downsample:  factor to downsample images.
        """
        assert images.shape[:-1] == masks.shape[:-1]
        assert images.shape[0] == index.shape[0]

        self.image_shape = images.shape[1:]
        self.mask_shape  = masks.shape[1:]

        self.images = images
        self.masks = masks
        self.index = index
        self.num_volumes = len(self.volumes())

        self.downsample(downsample)

        log.info(
            'Creating Data object with images of shape %s and %d volumes' % (str(self.images.shape), self.num_volumes))
        log.info('Images value range [%.1f, %.1f]' % (images.min(), images.max()))
        log.info('Masks value range [%.1f, %.1f]' % (masks.min(), masks.max()))

    def copy(self):
        return Data(np.copy(self.images), np.copy(self.masks), np.copy(self.index))

    def merge(self, other):
        assert self.images.shape[1:] == other.images.shape[1:], str(self.images.shape) + ' vs ' + str(
            other.images.shape)
        assert self.masks.shape[1:] == other.masks.shape[1:], str(self.masks.shape) + ' vs ' + str(other.masks.shape)

        log.info('Merging Data object of %d to this Data object of size %d' % (other.size(), self.size()))

        self.images = np.concatenate([self.images, other.images], axis=0)
        self.masks = np.concatenate([self.masks, other.masks], axis=0)
        self.index = np.concatenate([self.index, other.index], axis=0)
        self.num_volumes = len(self.volumes())

    def shuffle(self):
        idx = np.array(range(self.images.shape[0]))
        np.random.shuffle(idx)
        self.images = self.images[idx]
        self.masks = self.masks[idx]
        self.index = self.index[idx]

    def crop(self, shape):
        log.debug('Cropping images and masks to shape ' + str(shape))
        [images], [masks] = utils.data_utils.crop_same([self.images], [self.masks], size=shape, pad_mode='constant')
        self.images = images
        self.masks = masks
        assert self.images.shape[1:-1] == self.masks.shape[1:-1] == tuple(shape), \
            'Invalid shapes: ' + str(self.images.shape[1:-1]) + ' ' + str(self.masks.shape[1:-1]) + ' ' + str(shape)

    def volumes(self):
        return sorted(set(self.index))

    def get_images(self, vol):
        return self.images[self.index == vol]

    def get_masks(self, vol):
        return self.masks[self.index == vol]

    def size(self):
        return len(self.images)

    def sample_per_volume(self, num, seed=-1):
        log.info('Sampling %d from each volume' % num)
        if seed > -1:
            np.random.seed(seed)

        new_images, new_masks, new_scanner, new_index = [], [], [], []
        for vol in self.volumes():
            images = self.get_images(vol)
            masks = self.get_masks(vol)

            if images.shape[0] < num:
                log.debug('Volume %s contains less images: %d < %d. Sampling %d images.' %
                          (str(vol), images.shape[0], num, images.shape[0]))
                idx = range(len(images))
            else:
                idx = np.random.choice(images.shape[0], size=num, replace=False)

            images = np.array([images[i] for i in idx])
            masks = np.array([masks[i] for i in idx])
            index = np.array([vol] * num)

            new_images.append(images)
            new_masks.append(masks)
            new_index.append(index)

        self.images = np.concatenate(new_images, axis=0)
        self.masks = np.concatenate(new_masks, axis=0)
        self.index = np.concatenate(new_index, axis=0)

        log.info('Sampled %d images.' % len(self.images))

    def sample_images(self, num, seed=-1):
        log.info('Sampling %d images out of total %d' % (num, self.size()))
        if seed > -1:
            np.random.seed(seed)

        idx = np.random.choice(self.size(), size=num, replace=False)
        self.images = np.array([self.images[i] for i in idx])
        self.masks = np.array([self.masks[i] for i in idx])
        self.index = np.array([self.index[i] for i in idx])

    def get_sample_volumes(self, num, seed=-1):
        log.info('Sampling %d volumes out of total %d' % (num, self.num_volumes))
        if seed > -1:
            np.random.seed(seed)

        volumes = np.random.choice(self.volumes(), size=num, replace=False)
        return volumes

    def sample(self, num, seed=-1):
        if num == self.num_volumes:
            return

        volumes = self.get_sample_volumes(num, seed)
        self.filter_volumes(volumes)

    def filter_volumes(self, volumes):
        if len(volumes) == 0:
            self.images = np.array((0,) + self.images.shape[1:])
            self.masks = np.array((0,) + self.masks.shape[1:])
            self.index = np.array((0,) + self.index.shape[1:])
            self.num_volumes = 0
            return

        self.images = np.concatenate([self.get_images(v) for v in volumes], axis=0)
        self.masks = np.concatenate([self.get_masks(v) for v in volumes], axis=0)
        self.index = np.concatenate([self.index.copy()[self.index == v] for v in volumes], axis=0)
        self.num_volumes = len(volumes)

        log.info('Filtered volumes: %s of total %d images' % (str(volumes), self.size()))

    def shape(self):
        return self.image_shape

    def downsample(self, ratio=2):
        if ratio == 1: return

        self.images = block_reduce(self.images, block_size=(1, ratio, ratio, 1), func=np.mean)
        if self.masks is not None:
            self.masks = block_reduce(self.masks, block_size=(1, ratio, ratio, 1), func=np.mean)

        log.info('Downsampled data by %d to shape %s' % (ratio, str(self.images.shape)))

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        for i in range(self.images.shape[0]):
            np.savez_compressed(folder + '/images_%d' % i, self.images[i:i+1])
            np.savez_compressed(folder + '/masks_%d' % i, self.masks[i:i + 1])

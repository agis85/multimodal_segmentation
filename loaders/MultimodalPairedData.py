from loaders.data import Data
import utils.data_utils
import numpy as np
import logging
log = logging.getLogger('MultimodalPairedData')


class MultimodalPairedData(Data):
    """
    Container for multimodal data of image pairs. These are concatenated at the channel dimension
    """
    def __init__(self, images, masks, index, downsample=1):
        super(MultimodalPairedData, self).__init__(images, masks, index, downsample)
        self.num_modalities = self.images.shape[-1]
        self.masks_per_mod  = self.masks.shape[-1] // 2

        images_mod1 = self.images[..., 0:1]
        images_mod2 = self.images[..., 1:2]
        masks_mod1  = self.masks[..., 0:self.masks_per_mod]
        masks_mod2  = self.masks[..., self.masks_per_mod:]

        self.image_dict  = {0: images_mod1, 1: images_mod2}
        self.masks_dict  = {0: masks_mod1, 1: masks_mod2}

        del self.images
        del self.masks

    def get_images_modi(self, mod_i):
        return self.image_dict[mod_i]

    def get_masks_modi(self, mod_i):
        return self.masks_dict[mod_i]

    def set_images_modi(self, mod_i, images):
        self.image_dict[mod_i] = images

    def set_masks_modi(self, mod_i, masks):
        self.masks_dict[mod_i] = masks

    def get_volume_images_modi(self, mod_i, vol):
        return self.get_images_modi(mod_i)[self.index == vol]

    def get_volume_masks_modi(self, mod_i, vol):
        return self.get_masks_modi(mod_i)[self.index == vol]

    def filter_volumes(self, volumes):
        if len(volumes) == 0:
            for modi in range(self.num_modalities):
                self.set_images_modi(modi, np.array((0,) + self.image_shape))
                self.set_masks_modi(modi, np.array((0,) + self.mask_shape))
                self.index       = np.array((0,) + self.index.shape[1:])
                self.num_volumes = 0
            return

        for modi in range(self.num_modalities):
            self.set_images_modi(modi, np.concatenate([self.get_volume_images_modi(modi, v) for v in volumes], axis=0))
            self.set_masks_modi(modi,  np.concatenate([self.get_volume_masks_modi(modi, v)  for v in volumes], axis=0))

        self.index = np.concatenate([self.index.copy()[self.index == v] for v in volumes], axis=0)
        self.num_volumes = len(volumes)

        log.info('Filtered volumes: %s of total %d images' % (str(volumes), self.size()))

    def crop(self, shape):
        log.debug('Cropping images and masks to shape ' + str(shape))
        for modi in range(self.num_modalities):
            [images], [masks] = utils.data_utils.crop_same([self.get_images_modi(modi)], [self.get_masks_modi(modi)],
                                                           size=shape, pad_mode='constant')
            self.set_images_modi(modi, images)
            self.set_masks_modi(modi, masks)
            assert images.shape[1:-1] == masks.shape[1:-1] == tuple(shape), \
                'Invalid shapes: ' + str(images.shape[1:-1]) + ' ' + str(masks.shape[1:-1]) + ' ' + str(shape)

    def size(self):
        return np.max([self.get_images_modi(modi).shape[0] for modi in range(self.num_modalities)])

    def sample_images(self, num, seed=-1):
        log.info('Sampling %d images out of total %d' % (num, self.size()))
        if seed > -1:
            np.random.seed(seed)

        idx = np.random.choice(self.size(), size=num, replace=False)
        for modi in range(self.num_modalities):
            images = self.get_images_modi(modi)
            masks  = self.get_masks_modi(modi)

            self.set_images_modi(np.array([images[i] for i in idx]))
            self.set_masks_modi(np.array([masks[i] for i in idx]))
        self.index = np.array([self.index[i] for i in idx])

    def expand_pairs(self, offsets, mod_i, neighborhood=2):
        """
        Create more pairs by considering neighbour images. Change the object in-place
        :param offsets: number of neighbouring slices to pair with
        :param mod_i: which modality is enlarged
        :param neighborhood: the number of candidates.
        """
        assert mod_i in [0, 1], 'mod_i can be in [0, 1]. It defines the neighborhood of which modality to enlarge'
        log.debug('Enlarge neighborhood with %d pairs' % offsets)

        all_images, all_labels, all_index = [], [], []
        for vol in self.volumes():
            img_mod1 = self.get_volume_images_modi(mod_i, vol)
            img_mod2 = self.get_volume_images_modi(1 - mod_i, vol)

            num_images = img_mod2.shape[0]
            vol_img_mod1 = []
            for i in range(num_images):
                if img_mod1.shape[0] < 2 * offsets + 1:
                    value_range = list(range(0, img_mod1.shape[0])) + [0] * (2 * offsets + 1 - img_mod1.shape[0])
                elif i < offsets:
                    value_range = list(range(0, 2 * offsets + 1))
                elif i + offsets >= num_images:
                    value_range = list(range(num_images - (2 * offsets + 1), num_images))
                else:
                    value_range = list(range(i - offsets, i + offsets + 1))

                # rearrange values, such that the first value is the expertly paired one.
                value_range.insert(0, value_range.pop(value_range.index(i)))
                assert len(list(value_range)) == 2 * offsets + 1, \
                    'Invalid length: %d vs %d' % (2 * offsets + 1, len(list(value_range)))

                if len(value_range) > neighborhood:
                    new_value_range = [value_range[0]]
                    new_value_range += list(np.random.choice(value_range[1:], size=neighborhood - 1, replace=False))
                    value_range = new_value_range
                assert len(value_range) <= neighborhood, "Exceeded maximum neighborhood size"

                neighbour_imgs = np.concatenate([img_mod1[index:index+1] for index in value_range], axis=-1)
                vol_img_mod1.append(neighbour_imgs)

            all_images.append(np.concatenate(vol_img_mod1, axis=0))

        all_images = np.concatenate(all_images, axis=0)

        assert all_images.shape[-1] == neighborhood, '%s vs %s' % (all_images.shape[-1], neighborhood)

        if mod_i == 0:
            self.set_images_modi(0, all_images)
        elif mod_i == 1:
            self.set_images_modi(1, all_images)

    def randomise_pairs(self, length=3, seed=None):
        if seed is not None:
            np.random.seed(seed)
        log.debug('Randomising pairs within a volume')

        new_images, new_masks = [], []
        for vol in self.volumes():
            images = self.get_volume_images_modi(0, vol)
            masks  = self.get_volume_masks_modi(0, vol)

            offsets = np.random.randint(-length, length, size=images.shape[0])
            for off in range(length):
                if offsets[off] + off < 0:
                    offsets[off] = np.random.randint(-off, length, size=1)

            for i in range(1, length):
                if offsets[-i] + range(images.shape[0])[-i] >= images.shape[0]:
                    offsets[-i] = np.random.randint(-length, i, size=1)
            new_pair_index = np.array(range(images.shape[0])) + offsets

            new_images.append(images[new_pair_index])
            new_masks.append(masks[new_pair_index])

        self.set_images_modi(0, np.concatenate(new_images, axis=0))
        self.set_masks_modi(0, np.concatenate(new_masks, axis=0))

    def merge(self, other):
        log.info('Merging Data object of %d to this Data object of size %d' % (other.size(), self.size()))

        for mod in range(self.num_modalities):
            cur_img_mod = self.get_images_modi(mod)
            oth_img_mod = other.get_images_modi(mod)

            cur_msk_mod = self.get_masks_modi(mod)
            oth_msk_mod = other.get_masks_modi(mod)

            img_mod = np.concatenate([cur_img_mod, oth_img_mod], axis=0)
            msk_mod = np.concatenate([cur_msk_mod, oth_msk_mod], axis=0)

            self.set_images_modi(mod, img_mod)
            self.set_masks_modi(mod, msk_mod)

        self.index = np.concatenate([self.index, other.index], axis=0)
        assert self.get_images_modi(0).shape[0] == self.index.shape[0]

        self.num_volumes = len(self.volumes())
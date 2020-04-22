import logging
import os
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback

from costs import dice
from utils.image_utils import save_segmentation, intensity_augmentation

log = logging.getLogger('BaseSaveImage')


class BaseSaveImage(Callback):
    """
    Abstract base class for saving training images
    """

    def __init__(self, folder, model):
        super(BaseSaveImage, self).__init__()
        self.folder = os.path.join(folder, 'training_images')
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.model = model

    @abstractmethod
    def on_epoch_end(self, epoch=None, logs=None):
        pass


class SaveImage(Callback):
    """
    Simple callback that saves segmentation masks and dice error.
    """
    def __init__(self, folder, test_data, test_masks=None, input_len=None, comet_experiment=None):
        super(SaveImage, self).__init__()
        self.folder = folder
        self.test_data = test_data  # this can be a list of images of different spatial dimensions
        self.test_masks = test_masks
        self.input_len = input_len
        self.comet_experiment = comet_experiment

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        all_dice = []
        for i in range(len(self.test_data)):
            d, m = self.test_data[i], self.test_masks[i]
            s, im = save_segmentation(self.folder, self.model, d, m, 'slc_%d' % i)
            all_dice.append(-dice(self.test_masks[i:i+1], s))

            if self.comet_experiment is not None:
                plt.figure()
                plt.plot(0, 0)  # fake a line plot to upload to comet
                plt.imshow(im, cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                self.comet_experiment.log_figure(figure_name='segmentation', figure=plt)
                plt.close()

        f = open(os.path.join(self.folder, 'test_error.txt'), 'a+')
        f.writelines("%d, %.3f\n" % (epoch, np.mean(all_dice)))
        f.close()


class SaveEpochImages(Callback):
    def __init__(self, conf, model, img_gen, comet_experiment=None):
        super(SaveEpochImages, self).__init__()
        self.folder = conf.folder + '/training'
        self.conf = conf
        self.model = model
        self.gen = img_gen
        self.comet_experiment = comet_experiment
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def on_epoch_end(self, epoch, logs=None):
        x, m = next(self.gen)
        x = intensity_augmentation(x)

        y = self.model.predict(x)
        im1, im2 = save_multiimage_segmentation(x, m, y, self.folder, epoch)
        if self.comet_experiment is not None:
            plt.figure()
            plt.plot(0, 0)  # fake a line plot to upload to comet
            plt.imshow(im1, cmap='gray')
            plt.imshow(im2, cmap='gray', alpha=0.5)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            self.comet_experiment.log_figure(figure_name='segmentation', figure=plt)
            plt.close()


def save_multiimage_segmentation(x, m, y, folder, epoch):
    rows_img, rows_msk = [], []
    for i in range(x.shape[0]):
        if i == 4:
            break
        # y_list = [y[i, :, :, chn] for chn in range(y.shape[-1])]
        # m_list = [m[i, :, :, chn] for chn in range(m.shape[-1])]
        # if m.shape[-1] < y.shape[-1]:
        #     m_list += [np.zeros(shape=(m.shape[1], m.shape[2]))] * (y.shape[-1] - m.shape[-1])
        # assert len(y_list) == len(m_list), 'Incompatible sizes: %d vs %d' % (len(y_list), len(m_list))

        rows_img += [np.concatenate([x[i, :, :, 0], x[i, :, :, 0], x[i, :, :, 0]], axis=1)]
        rows_msk += [np.concatenate([np.zeros(x[i, :, :, 0].shape)] +
                            [sum([m[i, :, :, j] * (j + 1) * (1.0 / m.shape[-1]) for j in range(m.shape[-1])])] +
                            [sum([y[i, :, :, j] * (j + 1) * (1.0 / m.shape[-1]) for j in range(m.shape[-1])])], axis=1)]

    rows_img = np.concatenate(rows_img, axis=0)
    rows_msk = np.concatenate(rows_msk, axis=0)

    plt.figure()
    plt.imshow(rows_img, cmap='gray')
    plt.imshow(rows_msk, alpha=0.5)
    plt.savefig(folder + '/segmentations_epoch_%d.png' % (epoch))
    # scipy.misc.imsave(folder + '/segmentations_epoch_%d.png' % (epoch), im_plot)
    # return im_plot
    return rows_img, rows_msk


def get_s0chn(k, s):
    s_res = s.copy()
    chnk = s_res[..., k]
    # move channel k 1s to the background
    s_res[..., -1][chnk == 1] = 1
    s_res[..., k] = 0
    return s_res
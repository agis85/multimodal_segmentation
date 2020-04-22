import logging
import os
from abc import abstractmethod

from keras import Model, Input
from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from callbacks.image_callback import SaveImage
from costs import make_dice_loss_fnc, make_combined_dice_bce, weighted_cross_entropy_loss
from loaders import loader_factory

log = logging.getLogger('basenet')


class BaseNet(object):
    """
    Base model for segmentation neural networks
    """
    def __init__(self, conf):
        self.model = None
        self.conf = conf
        self.loader = None
        if hasattr(self.conf, 'dataset_name') and len(self.conf.dataset_name) > 0:
            self.loader = loader_factory.init_loader(self.conf.dataset_name)

    @abstractmethod
    def build(self):
        pass

    def load_models(self):
        pass

    @abstractmethod
    def get_segmentor(self, modality):
        pass

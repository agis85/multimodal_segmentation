import logging
from abc import abstractmethod


data_conf = {
    'chaos': '../../data/Chaos/MR',
}


class Loader(object):
    """
    Abstract class defining the behaviour of loaders for different datasets.
    """
    def __init__(self, volumes=None):
        self.num_masks   = 0
        self.num_volumes = 0
        self.input_shape = (None, None, 1)
        self.processed_folder = None
        if volumes is not None:
            self.volumes = volumes
        else:
            all_volumes = self.splits()[0]['training'] + self.splits()[0]['validation'] + self.splits()[0]['test']
            self.volumes = sorted(all_volumes)
        self.log = logging.getLogger('loader')

    @abstractmethod
    def splits(self):
        """
        :return: an array of splits into validation, test and train indices
        """
        pass

    @abstractmethod
    def load_all_modalities_concatenated(self, split, split_type, downsample):
        """
        Load multimodal data, and concatenate the images of the same volume/slice
        :param split:       the split number
        :param split_type:  training/validation/test
        :return:            a Data object of multimodal images
        """
        pass

    @abstractmethod
    def load_labelled_data(self, split, split_type, modality, normalise=True, downsample=1, root_folder=None):
        """
        Load labelled data.
        :param split:       the split number, e.g. 0, 1
        :param split_type:  the split type, e.g. training, validation, test, all (for all data)
        :param modality:    modality to load if the dataset has multimodal data
        :param normalise:   True/False: normalise images to [-1, 1]
        :param downsample:  downsample image ratio - used for for testing
        :param root_folder: root data folder
        :return:            a Data object containing the loaded data
        """
        pass

    @abstractmethod
    def load_unlabelled_data(self, split, split_type, modality, normalise=True, downsample=1):
        """
        Load unlabelled data.
        :param split:       the split number, e.g. 0, 1
        :param split_type:  the split type, e.g. training, validation, test, all (for all data)
        :param modality:    modality to load if the dataset has multimodal data
        :param normalise:   True/False: normalise images to [-1, 1]
        :return:            a Data object containing the loaded data
        """
        pass

    @abstractmethod
    def load_all_data(self, split, split_type, modality, normalise=True, downsample=1):
        """
        Load all images (labelled and unlabelled).
        :param split:       the split number, e.g. 0, 1
        :param split_type:  the split type, e.g. training, validation, test, all (for all data)
        :param modality:    modality to load if the dataset has multimodal data
        :param normalise:   True/False: normalise images to [-1, 1]
        :return:            a Data object containing the loaded data
        """
        pass

    def get_volumes_for_split(self, split, split_type):
        assert split_type in ['training', 'validation', 'test', 'all'], 'Unknown split_type: ' + split_type

        if split_type == 'all':
            volumes = sorted(self.splits()[split]['training'] + self.splits()[split]['validation'] +
                             self.splits()[split]['test'])
        else:
            volumes = self.splits()[split][split_type]
        return volumes

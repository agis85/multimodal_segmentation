from loaders.chaos import ChaosLoader


def init_loader(dataset):
    """
    Factory method for initialising data loaders by name.
    """
    if dataset == 'chaos':
        return ChaosLoader()
    return None

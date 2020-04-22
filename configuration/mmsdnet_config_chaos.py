from loaders import chaos

params = {
    'seed': 10,
    'folder': 'mmsdnet_chaos',
    'epochs': 500,
    'batch_size': 6,
    'split': 0,
    'dataset_name': 'chaos',
    'test_dataset': 'chaos',
    'input_shape': chaos.ChaosLoader().input_shape,
    'image_downsample': 1,                            # downsample image size: used for testing
    'modality': ['t1', 't2'],                         # list of [source, target] modalities
    'model': 'mmsdnet.MMSDNet',                       # model to load
    'executor': 'mmsdnet_executor.MMSDNetExecutor',   # model trainer
    'l_mix': 1,                                       # amount of supervision for target modality
    'decoder_type': 'film',                           # decoder type - can be film or spade
    'num_z': 8,                                       # dimensions of the modality factor
    'w_sup_M': 10,
    'w_adv_M': 1,
    'w_rec_X': 10,
    'w_adv_X': 1,
    'w_rec_Z': 1,
    'w_kl': 0.1,
    'lr': 0.0001,
}

# discriminator configs
d_mask_params  = {'filters': 4, 'lr': 0.0001, 'name': 'D_Mask', 'downsample_blocks': 4}

anatomy_encoder_params = {
    'normalise'   : 'batch',   # normalisation layer - can be batch or instance
    'downsample'  : 4,         # number of downsample layers of UNet encoder
    'filters'     : 64,        # number of filters in the first convolutional layer
    'out_channels': 8,         # number of output channels - dimensions of the anatomy factor
    'rounding'    : True
}


def get():
    shp = params['input_shape']
    ratio = params['image_downsample']
    shp = (int(shp[0] / ratio), int(shp[1] / ratio), shp[2])

    params['input_shape'] = shp
    params['num_masks'] = chaos.ChaosLoader().num_masks

    d_mask_params['input_shape'] = (shp[:-1]) + (chaos.ChaosLoader().num_masks,)

    anatomy_encoder_params['input_shape'] = shp
    anatomy_encoder_params['output_shape'] = (shp[:-1]) + (anatomy_encoder_params['out_channels'],)
    params.update({'anatomy_encoder': anatomy_encoder_params, 'd_mask_params': d_mask_params})
    return params

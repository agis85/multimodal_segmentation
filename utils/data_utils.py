import os
import numpy as np
import logging
log = logging.getLogger('data_utils')


def rescale(array, min_value=-1, max_value=1):
    """
    Rescales the input image between the min and max value.
    :param array:       a 4D array
    :param min_value:   the minimum value
    :param max_value:   the maximum value
    :return:            the rescaled array
    """
    if array.max() == array.min():
        array = (array * 0) + min_value
        return array
    array = (max_value - min_value) * (array - float(array.min())) / (array.max() - array.min()) + min_value
    assert array.max() == max_value and array.min() == min_value, '%d, %d' % (array.max(), array.min())
    return array

def normalise(image):
    """
    Normalise an image using the median and inter-quartile distance.
    :param image:   a 4D array
    :return:        the normalised image
    """
    array = image.copy()
    m = np.percentile(array, 50)
    s = np.percentile(array, 75) - np.percentile(array, 25)
    array = np.divide((array - m), s + 1e-12)

    assert not np.any(np.isnan(array)), 'NaN values in normalised array'
    return array


def crop_same(image_list, mask_list, size=(None, None), mode='equal', pad_mode='edge'):
    '''
    Crop the data in the image and mask lists, so that they have the same size.
    :param image_list: a list of images. Each element should be 4-dimensional, (sl,h,w,chn)
    :param mask_list:  a list of masks. Each element should be 4-dimensional, (sl,h,w,chn)
    :param size:       dimensions to crop the images to.
    :param mode:       can be one of [equal, left, right]. Denotes where to crop pixels from. Defaults to middle.
    :param pad_mode:   can be one of ['edge', 'constant']. 'edge' pads using the values of the edge pixels,
                       'constant' pads with a constant value
    :return:           the modified arrays
    '''
    min_w = np.min([m.shape[1] for m in mask_list]) if size[0] is None else size[0]
    min_h = np.min([m.shape[2] for m in mask_list]) if size[1] is None else size[1]

    # log.debug('Resizing list1 of size %s to size %s' % (str(image_list[0].shape), str((min_w, min_h))))
    # log.debug('Resizing list2 of size %s to size %s' % (str(mask_list[0].shape), str((min_w, min_h))))

    img_result, msk_result = [], []
    for i in range(len(mask_list)):
        im = image_list[i]
        m = mask_list[i]

        if m.shape[1] > min_w:
            m = _crop(m, 1, min_w, mode)
        if im.shape[1] > min_w:
            im = _crop(im, 1, min_w, mode)
        if m.shape[1] < min_w:
            m = _pad(m, 1, min_w, pad_mode)
        if im.shape[1] < min_w:
            im = _pad(im, 1, min_w, pad_mode)

        if m.shape[2] > min_h:
            m = _crop(m, 2, min_h, mode)
        if im.shape[2] > min_h:
            im = _crop(im, 2, min_h, mode)
        if m.shape[2] < min_h:
            m = _pad(m, 2, min_h, pad_mode)
        if im.shape[2] < min_h:
            im = _pad(im, 2, min_h, pad_mode)

        img_result.append(im)
        msk_result.append(m)
    return img_result, msk_result


def _crop(image, dim, nb_pixels, mode):
    diff = image.shape[dim] - nb_pixels
    if mode == 'equal':
        l = int(np.ceil(diff / 2))
        r = image.shape[dim] - l
    elif mode == 'right':
        l = 0
        r = nb_pixels
    elif mode == 'left':
        l = diff
        r = image.shape[dim]
    else:
        raise 'Unexpected mode: %s. Expected to be one of [equal, left, right].' % mode

    if dim == 1:
        return image[:, l:r, :, :]
    elif dim == 2:
        return image[:, :, l:r, :]
    else:
        return None


def _pad(image, dim, nb_pixels, mode='edge'):
    diff = nb_pixels - image.shape[dim]
    l = int(diff / 2)
    r = int(diff - l)
    if dim == 1:
        pad_width = ((0, 0), (l, r), (0, 0), (0, 0))
    elif dim == 2:
        pad_width = ((0, 0), (0, 0), (l, r), (0, 0))
    else:
        return None

    if mode == 'edge':
        new_image = np.pad(image, pad_width, 'edge')
    elif mode == 'constant':
        new_image = np.pad(image, pad_width, 'constant', constant_values=np.min(image))
    else:
        raise Exception('Invalid pad mode: ' + mode)

    return new_image


def sample(data, nb_samples, seed=-1):
    if seed > -1:
        np.random.seed(seed)
    idx = np.random.choice(len(data), size=nb_samples, replace=False)
    return np.array([data[i] for i in idx])

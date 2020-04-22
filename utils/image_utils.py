
import os

import albumentations.augmentations.transforms
import matplotlib.path as pth
import numpy as np
from PIL import Image, ImageDraw
from scipy.misc import imsave
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation

import utils.data_utils


def save_segmentation(folder, model, images, masks, name_prefix):
    '''
    :param folder: folder to save the image
    :param model : segmentation model
    :param images: an image of shape [H,W,chn]
    :param masks : a mask of shape [H,W,chn]
    :return      : the predicted segmentation mask
    '''
    images = np.expand_dims(images, axis=0)
    masks  = np.expand_dims(masks, axis=0)
    s = model.predict(images)

    # In this case the segmentor is multi-output, with each output corresponding to a mask.
    if len(s[0].shape) == 4:
        s = np.concatenate(s, axis=-1)

    mask_list_pred = [s[:, :, :, j:j + 1] for j in range(s.shape[-1])]
    mask_list_real = [masks[:, :, :, j:j + 1] for j in range(masks.shape[-1])]
    if masks.shape[-1] < s.shape[-1]:
        mask_list_real += [np.zeros(shape=masks.shape[0:3] + (1,))] * (s.shape[-1] - masks.shape[-1])

    # if we use rotations, the sizes might differ
    m1, m2 = utils.data_utils.crop_same(mask_list_real, mask_list_pred)
    images_cropped, _ = utils.data_utils.crop_same([images], [images.copy()], size=(m1[0].shape[1], m1[0].shape[2]))
    mask_list_real = [s[0, :, :, 0] for s in m1]
    mask_list_pred = [s[0, :, :, 0] for s in m2]
    images_cropped = [s[0, :, :, 0] for s in images_cropped]

    row1 = np.concatenate(images_cropped + mask_list_pred, axis=1)
    row2 = np.concatenate(images_cropped + mask_list_real, axis=1)
    im = np.concatenate([row1, row2], axis=0)
    imsave(os.path.join(folder, name_prefix + '.png'), im)
    return s, im


def makeTextHeaderImage(col_widths, headings, padding=(5, 5)):
    im_width = len(headings) * col_widths
    im_height = padding[1] * 2 + 11

    img = Image.new('RGB', (im_width, im_height), (0, 0, 0))
    d = ImageDraw.Draw(img)

    for i, txt in enumerate(headings):

        while d.textsize(txt)[0] > col_widths - padding[0]:
            txt = txt[:-1]
        d.text((col_widths * i + padding[0], + padding[1]), txt, fill=(1, 0, 0))

    raw_img_data = np.asarray(img, dtype="int32")

    return raw_img_data[:, :, 0]


def process_contour(segm_mask, endocardium, epicardium=None):
    '''
    in each pixel we sample these 8 points:
     _________________
    |    *        *   |
    |  *            * |
    |                 |
    |                 |
    |                 |
    |  *            * |
    |    *        *   |
     ------------------
    we say a pixel is in the contour if half or more of these 8 points fall within the contour line
    '''

    contour_endo = pth.Path(endocardium, closed=True)
    contour_epi = pth.Path(epicardium, closed=True) if epicardium is not None else None
    for x in range(segm_mask.shape[1]):
        for y in range(segm_mask.shape[0]):
            for (dx, dy) in [(-0.25, -0.375), (-0.375, -0.25), (-0.25, 0.375), (-0.375, 0.25), (0.25, 0.375),
                             (0.375, 0.25), (0.25, -0.375), (0.375, -0.25)]:

                point = (x + dx, y + dy)
                if contour_epi is None and contour_endo.contains_point(point):
                    segm_mask[y, x] += 1
                elif contour_epi is not None and \
                        contour_epi.contains_point(point) and not contour_endo.contains_point(point):
                    segm_mask[y, x] += 1

    segm_mask = (segm_mask >= 4) * 1.
    return segm_mask


def intensity_augmentation(batch):
    """
    Perform a random intensity augmentation
    :param batch: an image batch (B,H,W,C)
    :return:      the intensity augmented batch
    """
    aug = albumentations.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.01,
                                                                           contrast_limit=(0.99, 1.01))
    batch = utils.data_utils.rescale(batch, 0, 1)
    batch = aug(image = batch)['image']
    return utils.data_utils.rescale(batch, -1, 1)

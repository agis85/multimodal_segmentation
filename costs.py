import logging

import numpy as np
from keras import backend as K
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform

log = logging.getLogger()

lambda_bce = 0.01

########## RECONSTRUCTION LOSSES ##########

def make_similarity_weighted_mae(weights):
    def similarity_weighted_mae(y_true, y_pred):
        shape = K.int_shape(y_pred)
        w_reshaped = tf.expand_dims(tf.expand_dims(weights, axis=1), axis=1)
        mae = tf.multiply(K.abs(y_true - y_pred), tf.tile(w_reshaped, (1, shape[1], shape[2], 1)))
        return K.mean(mae)

    return similarity_weighted_mae


def mae_single_input(y):
    y1, y2 = y
    return K.mean(K.abs(y1-y2), axis=(1, 2))


########## SEGMENTATION LOSSES ##########

def dice(y_true, y_pred, binarise=False, smooth=1e-12):
    y_pred = y_pred[..., 0:y_true.shape[-1]]

    # Cast the prediction to binary 0 or 1
    if binarise:
        y_pred = np.round(y_pred)

    # Symbolically compute the intersection
    y_int = y_true * y_pred
    return np.mean((2 * np.sum(y_int, axis=(1, 2, 3)) + smooth)
                   / (np.sum(y_true, axis=(1, 2, 3)) + np.sum(y_pred, axis=(1, 2, 3)) + smooth))

def dice_coef_perbatch(y_true, y_pred):
    # Symbolically compute the intersection
    intersection = K.sum(y_true * y_pred, axis=(1, 2, 3))
    union = K.sum(y_true, axis=(1, 2, 3)) + K.sum(y_pred, axis=(1, 2, 3))
    dice = (2 * intersection + 1e-12) / (union + 1e-12)
    return 1 - dice

def dice_coef_loss(y_true, y_pred):
    '''
    DICE Loss.
    :param y_true: a tensor of ground truth data
    :param y_pred: a tensor of predicted data
    '''
    return K.mean(dice_coef_perbatch(y_true, y_pred), axis=0)


def make_dice_loss_fnc(restrict_chn=1):
    log.debug('Making DICE loss function for the first %d channels' % restrict_chn)

    def dice_fnc(y_true, y_pred):
        y_pred_new = y_pred[..., 0:restrict_chn] + 0.
        y_true_new = y_true[..., 0:restrict_chn] + 0.
        return dice_coef_loss(y_true_new, y_pred_new)

    return dice_fnc


def weighted_cross_entropy_loss(y_pred, y_true):
    """
    Define weighted cross - entropy function for classification tasks.
    :param y_pred: tensor[None, width, height, n_classes]
    :param y_true: tensor[None, width, height, n_classes]
    """
    num_classes = K.int_shape(y_true)[-1]
    n = [tf.reduce_sum(tf.cast(y_true[..., c], tf.float32)) for c in range(num_classes)]
    n_tot = tf.reduce_sum(n)
    weights = [n_tot / (n[c] + 1e-12) for c in range(num_classes)]
    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))
    w_cross_entropy = tf.multiply(y_true * tf.log(y_pred + 1e-12), weights)
    w_cross_entropy = -tf.reduce_sum(w_cross_entropy, reduction_indices=[1])
    loss = tf.reduce_mean(w_cross_entropy, name='weighted_cross_entropy')
    return loss


def weighted_cross_entropy_perbatch(y_pred, y_true):
    """
    Define weighted cross - entropy function for classification tasks.
    :param y_pred: tensor[None, width, height, n_classes]
    :param y_true: tensor[None, width, height, n_classes]
    """
    shape = K.int_shape(y_true)
    restrict_chn = shape[-1]

    n = tf.reduce_sum(y_true, axis=[0, 1, 2])
    n_tot = tf.reduce_sum(n, axis=0)
    weights = n_tot / (n + 1e-12)

    y_pred = tf.reshape(y_pred, (-1, shape[1] * shape[2], restrict_chn))
    y_true2 = tf.to_float(tf.reshape(y_true, (-1, shape[1] * shape[2], restrict_chn)))
    softmax = tf.nn.softmax(y_pred)

    w_cross_entropy = -tf.reduce_sum(y_true2 * tf.log(softmax + 1e-12) * weights, reduction_indices=[2])
    # w_cross_entropy = tf.multiply(w_cross_entropy, tf.tile(tf.expand_dims(contributions, axis=-1), (1, shape[1] * shape[2])))
    loss = tf.reduce_mean(w_cross_entropy, axis=1, name='softmax_weighted_cross_entropy')
    return loss


def similarity_weighted_dice(weights, restrict_chn):
    log.debug('Making similarity weighted DICE loss function for the first %d channels' % restrict_chn)

    def weighted_dice_fnc(y_true):
        y_pred_new, y_true_new = y_true
        # assert K.int_shape(y_pred)[-1] == K.int_shape(y_true)[-1] + 1, 'y_pred does not contain similarity weights'

        y_pred_new = y_pred_new[..., 0:restrict_chn] + 0.
        y_true_new = y_true_new[..., 0:restrict_chn] + 0.

        intersection = K.sum(y_true_new * y_pred_new, axis=(1, 2, 3))
        union = K.sum(y_true_new, axis=(1, 2, 3)) + K.sum(y_pred_new, axis=(1, 2, 3))
        dice = (2 * intersection + 1e-5) / (union + 1e-5)
        return K.mean(weights * (1 - dice))

    return weighted_dice_fnc


def make_combined_dice_bce(num_classes):
    dice = make_dice_loss_fnc(num_classes)
    bce = weighted_cross_entropy_loss

    def combined_dice_bce(y_true, y_pred):
        return dice(y_true, y_pred) + lambda_bce * bce(y_true, y_pred)

    return combined_dice_bce

def make_combined_dice_bce_perbatch(num_classes):
    def fnc(y_true, y_pred):
        y_pred_new = y_pred[..., 0:num_classes] + 0.
        y_true_new = y_true[..., 0:num_classes] + 0.
        return dice_coef_perbatch(y_true_new, y_pred_new) + lambda_bce * weighted_cross_entropy_perbatch(y_true, y_pred)
    return fnc

def similarity_weighted_dice_bce(contributions, restrict_chn, eps=1e-5):
    log.debug('Making similarity weighted DICE loss function for the first %d channels' % restrict_chn)

    def weighted_dice_fnc(y_true, y_pred):
        y_pred_new = y_pred[..., 0:restrict_chn] + 0.
        y_true_new = y_true[..., 0:restrict_chn] + 0.

        intersection = K.sum(y_true_new * y_pred_new, axis=(1, 2, 3))
        union = K.sum(y_true_new, axis=(1, 2, 3)) + K.sum(y_pred_new, axis=(1, 2, 3))
        dice = (2 * intersection + eps) / (union + eps)
        return K.mean(contributions * (1 - dice))

    def weighted_cross_entropy(y_pred, y_true):
        """
        Define weighted cross - entropy function for classification tasks.
        :param y_pred: tensor[None, width, height, n_classes]
        :param y_true: tensor[None, width, height, n_classes]
        """
        shape = K.int_shape(y_true)
        num_chn = shape[-1]

        n = tf.reduce_sum(y_true, axis=[0, 1, 2])
        n_tot = tf.reduce_sum(n, axis=0)
        weights = n_tot / (n + eps)

        y_pred = tf.reshape(y_pred, (-1, shape[1] * shape[2], num_chn))
        y_true2 = tf.to_float(tf.reshape(y_true, (-1, shape[1] * shape[2], num_chn)))

        w_cross_entropy = -tf.reduce_sum(y_true2 * tf.log(y_pred + eps) * weights, reduction_indices=[2])
        w_cross_entropy = tf.multiply(w_cross_entropy, tf.tile(contributions, (1, shape[1] * shape[2])))
        loss = tf.reduce_mean(w_cross_entropy, name='weighted_cross_entropy')

        return loss

    def combined_fnc(y_true, y_pred):
        return weighted_dice_fnc(y_true, y_pred)  + lambda_bce * weighted_cross_entropy(y_true, y_pred)

    return combined_fnc

########## VAE LOSSES ##########

def kl(args):
    mean, log_var = args
    kl_loss = -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
    return K.reshape(kl_loss, (-1, 1))


########## OTHER LOSSES ##########

def ypred(y_true, y_pred):
    return y_pred


def distance_correlation(A, B):
    '''
    Calculate the Distance Correlation between the two vectors. https://en.wikipedia.org/wiki/Distance_correlation
    Value of 0 implies independence. A and B can be vectors of different length.
    :param A:    vector A of shape (num_samples, sizeA)
    :param B:    vector B of shape (num_samples, sizeB)
    :return:     the distance correlation between A and B
    '''
    n = A.shape[0]
    if B.shape[0] != A.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(A))
    b = squareform(pdist(B))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

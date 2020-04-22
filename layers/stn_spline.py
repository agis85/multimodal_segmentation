import tensorflow as tf
from keras import Input, Model
from keras.engine import Layer
from keras.layers import Concatenate, MaxPooling2D, Conv2D, Flatten, Dense, Reshape, LeakyReLU

from layers.interpolate_spline import interpolate_spline

bilinear_interpolation = tf.contrib.resampler.resampler
import numpy as np
import logging
log = logging.getLogger('stn_spline')


class ThinPlateSpline2D(Layer):
    """
    Keras layer for Thin Plate Spline interpolation.
    """
    def __init__(self, input_volume_shape, cp_dims, num_channels, inverse=False, order=2, **kwargs):

        self.vol_shape = input_volume_shape
        self.data_dimensionality = len(input_volume_shape)
        self.cp_dims = cp_dims
        self.num_channels = num_channels
        self.initial_cp_grid = None
        self.flt_grid = None
        self.inverse = inverse
        self.order = order
        super(ThinPlateSpline2D, self).__init__(**kwargs)

    def build(self, input_shape):

        self.initial_cp_grid = nDgrid(self.cp_dims)
        self.flt_grid = nDgrid(self.vol_shape)

        super(ThinPlateSpline2D, self).build(input_shape)

    # @tf.contrib.eager.defun
    def interpolate_spline_batch(self, cp_offsets_single_batch):

        warped_cp_grid = self.initial_cp_grid + cp_offsets_single_batch

        if self.inverse:
            interpolated_sample_locations = interpolate_spline(train_points=warped_cp_grid,
                                                               train_values=self.initial_cp_grid,
                                                               query_points=self.flt_grid,
                                                               order=self.order)
        else:
            interpolated_sample_locations = interpolate_spline(train_points=self.initial_cp_grid,
                                                               train_values=warped_cp_grid,
                                                               query_points=self.flt_grid,
                                                               order=self.order)

        return interpolated_sample_locations

    def call(self, args):

        vol, cp_offsets = args

        interpolated_sample_locations = tf.map_fn(self.interpolate_spline_batch, cp_offsets)[:, 0]

        interpolated_sample_locations = tf.reverse(interpolated_sample_locations, axis=[-1])

        interpolated_sample_locations = tf.multiply(interpolated_sample_locations,
                                                    [self.vol_shape[1] - 1, self.vol_shape[0] - 1])
        warped_volume = bilinear_interpolation(vol, interpolated_sample_locations)
        warped_volume = tf.reshape(warped_volume, (-1,) + tuple(self.vol_shape) + (self.num_channels,))
        return warped_volume


def nDgrid(dims, normalise=True, center=False, dtype='float32'):
    '''
    returns the co-ordinates for an n-dimentional grid as a (num-points, n) shaped array
    e.g. dims=[3,3] would return:
    [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    if not normalized == False, or:
    [[0,0],[0,0.5],[0,1.],[0.5,0],[0.5,0.5],[0.5,1.],[1.,0],[1.,0.5],[1.,1.]]
    if normalized == True.
    '''
    if len(dims) == 2:
        grid = np.expand_dims(np.mgrid[:dims[0], :dims[1]].reshape((2, -1)).T, 0)

    if len(dims) == 3:
        grid = np.expand_dims(np.mgrid[:dims[0], :dims[1], :dims[2]].reshape((3, -1)).T, 0)

    if normalise == True:
        grid = grid / (1. * (np.array([[dims]]) - 1))

        if center == True:
            grid = (grid - 1) * 2

    return tf.cast(grid, dtype=dtype)


def build_locnet(input_shape1, input_shape2, output_shape):
    """
    Build STN for calculating the parameters of STN.
    :param input_shape1: shape of input tensor 1
    :param input_shape2: shape of input tensor 2
    :param output_shape: number of control points to predict
    :return: a Keras model
    """
    input1 = Input(shape=input_shape1)
    input2 = Input(shape=input_shape2)
    stacked = Concatenate()([input1, input2])

    l = Conv2D(20, 5)(stacked)
    l = LeakyReLU()(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)
    l = Conv2D(20, 5)(l)
    l = LeakyReLU()(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)
    l = Conv2D(20, 5)(l)
    l = LeakyReLU()(l)
    l = Flatten()(l)
    l = Dense(100, activation='tanh')(l)
    theta = Dense(output_shape, kernel_initializer='zeros', bias_initializer='zeros')(l)
    theta = Reshape((int(output_shape / 2), 2))(theta)
    m = Model(input=[input1, input2], output=theta, name='stn_locnet')
    m.summary(print_fn=log.info)
    return m

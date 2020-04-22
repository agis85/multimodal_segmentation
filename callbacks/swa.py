#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic Weight Averaging: https://arxiv.org/abs/1803.05407
Keras implementation adapted from https://github.com/kristpapadopoulos/keras-stochastic-weight-averaging.
"""

import logging
import keras

log = logging.getLogger('swa')


class SWA(keras.callbacks.Callback):
    def __init__(self, swa_epoch, model_build_fnc, build_params):
        super(SWA, self).__init__()
        self.swa_epoch = swa_epoch
        self.model_build_fnc = model_build_fnc
        self.build_params = build_params
        self.clone = None

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))

    def on_epoch_end(self, epoch, logs=None):
        if epoch <= self.swa_epoch:
            self.swa_weights = self.model.get_weights()

        elif epoch > self.swa_epoch:
            for i, layer in enumerate(self.swa_weights):
                self.swa_weights[i] = (self.swa_weights[i] * (epoch - self.swa_epoch) + self.model.get_weights()[i])\
                                      / ((epoch - self.swa_epoch) + 1)

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        log.debug('Final model parameters set to stochastic weight average.')

    def get_clone_model(self):
        if self.clone is None:
            if self.build_params is not None:
                self.clone = self.model_build_fnc(self.build_params)
            else:
                self.clone = self.model_build_fnc()
        self.clone.set_weights(self.swa_weights)
        return self.clone

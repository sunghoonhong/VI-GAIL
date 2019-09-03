import os
import csv
import time
import argparse
import numpy as np

import cv2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Activation
from tensorflow.keras.layers import MaxPooling2D, Flatten, Concatenate
from tensorflow.keras.layers import BatchNormalization as Bn
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from models.Global_Encoder import Encoder
from utils.config import * 


class DiscretePosterior(Model):
    '''
        (s_t, a_(t-1), c_(t-1)) -> p(c_t).
        It will be used to approximate prior
    '''
    def __init__(self, lr):
        super(DiscretePosterior, self).__init__()
        self.opt = Adam(learning_rate=lr)
        self.encoder = Encoder()
        self.concat = Concatenate()

        # hidden
        self.hiddens = [
            Dense(128, activation='elu', kernel_initializer='he_normal', use_bias=False), Bn(),
            Dense(128, activation='elu', kernel_initializer='he_normal', use_bias=False), Bn()
        ]

        # output
        self.out = Dense(DISC_CODE_NUM, activation='softmax', kernel_initializer=tf.random_uniform_initializer(minval=-1e-3, maxval=1e-3))

    def call(self, s, a, prev_c):
        s = self.encoder(s)  # (N, 128)
        x = self.concat([s, a, prev_c])
        for layer in self.hiddens:
            x = layer(x)
        out = self.out(x)
        return out


class Actor(Model):
    def __init__(self, lr):
        super(Actor, self).__init__()
        self.opt = Adam(learning_rate=lr)
        self.encoder = Encoder()
        self.concat = Concatenate()

        # hidden
        self.hiddens = [
            Dense(128, activation='elu', kernel_initializer='he_normal', use_bias=False), Bn(),
            Dense(128, activation='elu', kernel_initializer='he_normal', use_bias=False), Bn()
        ]

        # output
        self.out = Dense(ACTION_NUM, activation='softmax', kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, s, c):
        s = self.encoder(s)
        x = self.concat([s, c])
        for layer in self.hiddens:
            x = layer(x)
        policy = self.out(x)
        return policy
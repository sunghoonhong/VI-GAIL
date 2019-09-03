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


class Critic(Model):
    def __init__(self, lr):
        super(Critic, self).__init__()
        self.opt = Adam(learning_rate=lr)
        self.encoder = Encoder()
        # hidden
        self.hiddens = [
            Dense(128, activation='elu', kernel_initializer='he_normal', use_bias=False), Bn(),
            Dense(128, activation='elu', kernel_initializer='he_normal', use_bias=False), Bn()
        ]

        # output
        self.value = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, s):
        x = self.encoder(s)
        for layer in self.hiddens:
            x = layer(x)
        value = self.value(x)
        return value


class Discriminator(Model):
    def __init__(self, lr):
        super(Discriminator, self).__init__()
        self.opt = Adam(learning_rate=lr)
        self.encoder = Encoder()
        self.concat = Concatenate()
        # hidden
        self.hiddens = [
            Dense(128, activation='elu', kernel_initializer='he_normal', use_bias=False), Bn(),
            Dense(128, activation='elu', kernel_initializer='he_normal', use_bias=False), Bn()
        ]
        # output
        self.out = Dense(1, activation='sigmoid', kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))   # 0: expert, 1: agent

    def call(self, s, a):
        s = self.encoder(s)
        x = self.concat([s, a])
        for layer in self.hiddens:
            x = layer(x)
        out = self.out(x)
        return out
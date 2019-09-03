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
from utils.config import *


class Actor(Model):
    def __init__(self, lr, hidden_units):
        super(Actor, self).__init__()
        self.opt = Adam(learning_rate=lr)

        self.concat = Concatenate()
        # hidden
        self.hiddens = [Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]

        # output
        self.out = Dense(ACTION_NUM, activation='softmax', kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, encoder, s, c):
        s = encoder.forward(s)
        x = self.concat([s, c])
        for layer in self.hiddens:
            x = layer(x)
        policy = self.out(x)
        return policy


class Critic(Model):
    def __init__(self, lr, hidden_units):
        super(Critic, self).__init__()
        self.opt = Adam(learning_rate=lr)

        # hidden
        self.hiddens = [Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]

        # output
        self.value = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, encoder, s):
        x = encoder.forward(s)
        for layer in self.hiddens:
            x = layer(x)
        value = self.value(x)
        return value


class Discriminator(Model):
    def __init__(self, lr, hidden_units, reduce_units):
        super(Discriminator, self).__init__()
        self.opt = Adam(learning_rate=lr)
        
        self.concat = Concatenate()
        self.reduces = [Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in reduce_units]
        # hidden
        self.hiddens = [Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]

        # output
        self.out = Dense(1, activation='sigmoid', kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))   # 0: expert, 1: agent

    def call(self, encoder, s, a):
        s = encoder.forward(s)
        for layer in self.reduces:
            s = layer(s)
        x = self.concat([s, a])
        for layer in self.hiddens:
            x = layer(x)
        out = self.out(x)
        return out
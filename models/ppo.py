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
    def __init__(self, lr):
        super(Actor, self).__init__()
        self.opt = Adam(learning_rate=lr)
        # hidden
        self.hiddens = [
            Dense(256, activation='relu')
        ]

        # output
        self.policy = Dense(ACTION_NUM, activation='softmax', kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, encoder, s):
        x = encoder(s)
        for layer in self.hiddens:
            x = layer(x)
        action = self.policy(x)
        return action


class Critic(Model):
    def __init__(self, lr):
        super(Critic, self).__init__()
        self.opt = Adam(learning_rate=lr)
        # hidden
        self.hiddens = [
            Dense(256, activation='relu')
        ]

        # output
        self.value = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, encoder, s):
        x = encoder(s)
        for layer in self.hiddens:
            x = layer(x)
        value = self.value(x)
        return value
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from utils.config import *


class Encoder(Model):
    def __init__(self, lr, hidden_units):
        super(Encoder, self).__init__()
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.conv_layers = [    # (32, 32, 3)
            Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), activation='relu', kernel_initializer='he_normal'),  # (15, 15, 16)
            Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_normal'),  # (7, 7, 32)
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal')                  # (5, 5, 32)
        ]
        self.flatten = Flatten()    # (800,)
        self.hiddens = [Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]

    def __call__(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.hiddens:
            x = layer(x)
        return x
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, PReLU
from tensorflow.keras.models import Model
from utils.config import *
from utils.util import *



class VAE_Encoder(Model):
    def __init__(self, latent_num):
        super(VAE_Encoder, self).__init__()
        self.latent_num = latent_num
        self.conv_layers = [    # (32, 32, 3)
            Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), activation='relu'),    # (16, 16, 32)
            Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation='relu'),    # (6, 6, 64)
            Conv2D(filters=128, kernel_size=(3,3), activation='relu'),                  # (4, 4, 128)
            Conv2D(filters=256, kernel_size=(3,3), activation='relu'),                  # (2, 2, 256)
        ]
        self.flatten = Flatten()
        self.mean_out = Dense(latent_num)
        self.std_out = Dense(latent_num, activation='sigmoid')

    def __call__(self, x, sampling=False):
        #returns out, mean, std
        m, s = self.encode(x)
        if sampling:
            noise = np.random.normal(size=self.latent_num)
            z = m + s * noise
        else:
            z = m
        return z, m, s
    
    def forward(self, x):
        m, _ = self.encode(x)
        return m

    def encode(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        m = self.mean_out(x)
        s = self.std_out(x)
        return m, s

    def set_weights_by_list(self, l):
        for i, layer in enumerate(self.conv_layers):
            layer.set_weights(l[i])
        self.mean_out.set_weights(l[-2])
        self.std_out.set_weights(l[-1])

    def save(self, dir, name):
        self.save_weights(dir + name + '.h5')


class VAE_Network(Model):
    def __init__(self):
        super(VAE_Network, self).__init__()
        self.encoder = VAE_Encoder(latent_num=VAE_CNN_LATENT)
        self.process_out = Dense(VAE_HIDDEN_SIZE)
        self.prelu = PReLU()    # (1, 1, 1024)
        self.deconv_layers = [
            Conv2DTranspose(filters=128, kernel_size=(4,4), activation='relu'),                 # (4, 4, 128)
            Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(1,1), activation='relu'),   # (7, 7, 64) 
            Conv2DTranspose(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu'),   # (15, 15, 32)
            Conv2DTranspose(filters=3, kernel_size=(4,4), strides=(2,2), activation='sigmoid'), # (32, 32, 3)
        ]
        self.opt = tf.optimizers.Adam(learning_rate=VAE_LR)

    def __call__(self, x, sampling=True):
        #returns out, mean, std
        x, m, s = self.encoder(x, sampling)
        x = self.prelu(self.process_out(x))
        x = tf.reshape(x, [-1, 1, 1, VAE_HIDDEN_SIZE])
        for layer in self.deconv_layers:
            x = layer(x)
        return x, m, s

    def update(self, data):
        #data should have size of (batch_size, 64, 64, 4)
        with tf.GradientTape() as tape:
            out, m, s = self.__call__(data)
            reconstruction_loss = tf.reduce_mean(tf.sqrt(tf.abs(data-out)+1e-16))  #tf.sqrt(tf.sqrt((data - out)**2)))
            regularization = tf.reduce_mean(tf_gaussian_KL(m, 0, s, 1))
            loss = reconstruction_loss + VAE_BETA*regularization
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.opt.apply_gradients(zip(gradients, self.trainable_variables))
        return loss.numpy()

    def get_encoder_weights(self):
        weights = []
        for layer in self.conv_layers:
            weights.append(layer.get_weights())
        weights.append(self.mean_out.get_weights())
        weights.append(self.std_out.get_weights())
        return weights

    def save(self, dir, name):
        self.save_weights(dir + name + '.h5')

    def load(self, dir, name):
        self.load_weights(dir + name + '.h5')
        print('successfully loaded vae weight')




if __name__ == '__main__':
    dummy = np.zeros(shape=(1,84,84,4))
    vae = VAE_Network()
    encoder = VAE_Encoder()
    encoder(dummy, True)
    vae(dummy)
    vae.load('vae')
    weights = vae.get_encoder_weights()
    encoder.set_weights_by_list(weights)
    encoder.save('vae_encoder')





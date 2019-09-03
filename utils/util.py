import cv2
import numpy as np
import tensorflow as tf
from utils.config import * 


def one_hot_batch(arr, idx):
    arr = np.eye(len(arr), dtype=np.float32)[[idx]]
    return arr

def preprocess_obs(obs):
    '''
        (1, 32, 32, 3)
    '''
    obs = cv2.resize(obs, OBS_RESIZE) # (32, 32, 3)
    obs = np.reshape(obs, [-1] + OBS_SHAPE).astype(np.float32)
    obs /= 255.
    return obs

# Discrete Space
def tf_sample_gumbel(shape, eps=1e-20):
    U = tf.random.uniform(shape=shape, dtype=tf.float32)
    return - tf.math.log(-tf.math.log(U + eps) + eps)

def tf_gumbel_softmax_sample(x, temperature):
    y = x + tf_sample_gumbel(x.shape)
    y = tf.math.softmax(y / temperature)
    return y

def tf_reparameterize(x, temperature):
    '''
        return tensor 
    '''
    return tf_gumbel_softmax_sample(x, temperature)

def tf_gaussian_KL(m1, m2, s1, s2):
    #calculating two gaussian (m1, s1), (m2, s2)
    g = tf.math.log(s2/(s1+1e-8)) + tf.divide(s1**2 + (m1 - m2)**2, 2*(s2**2)) - 0.5
    return g
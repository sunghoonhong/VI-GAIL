import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0' # '1', '2', '3', '0,1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import csv
import time
import math
import random
import argparse
import numpy as np
# import gym

import cv2
import tensorflow as tf

from env.navigate import NaviEnv
from models.gail import Actor
# from models.Global_Encoder import Encoder
from models.VAE_Network import VAE_Encoder
from utils.memory import HorizonMemory, ReplayMemory
from utils.util import *
from utils.config import *


class BC:
    def __init__(self, global_norm, actor_lr, actor_units, 
                epochs, batch_size, data_dir, demo_list):
        # build network
        self.actor  = Actor(lr=actor_lr, hidden_units=actor_units)
        self.encoder = VAE_Encoder(latent_num=64)
        self.opt = tf.keras.optimizers.Adam(learning_rate=actor_lr)

        # set hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_global_norm = global_norm

        # build expert demonstration Pipeline
        self.data_dir = data_dir
        self.demo_list = os.listdir(data_dir)
        self.demo_group_num = 1000
        self.demo_rotate = 20
        assert len(demo_list) >= self.demo_group_num
        self.set_demo()

        # ready
        self.dummy_forward()
        self.vars = self.actor.trainable_variables + self.encoder.trainable_variables

    def dummy_forward(self):
        # connect networks
        dummy_state = np.zeros([1] + STATE_SHAPE, dtype=np.float32)
        self.encoder(dummy_state)
        self.actor(self.encoder, dummy_state)

    def set_demo(self):
        self.demo_list = os.listdir(data_dir)
        selected_demos = random.sample(self.demo_list, self.demo_group_num)

        expert_states = []
        expert_actions = []
        for demo_name in selected_demos:
            demo = np.load(self.data_dir + demo_name)
            states = demo['state']
            actions = demo['action']
    
            expert_states.append(states)
            expert_actions.append(actions)
        self.expert_states = np.concatenate(expert_states, axis=0)
        self.expert_actions = np.concatenate(expert_actions, axis=0)
        del demo

    def get_demonstration(self, sample_num):
        index = np.arange(len(self.expert_states))
        try:
            assert len(self.expert_states) >= sample_num
        except Exception:
            self.set_demo()
        np.random.shuffle(index)
        index = index[:sample_num]
        return self.expert_states[index], self.expert_actions[index]

    def update(self):
        # load expert demonstration
        s_e, a_e = self.get_demonstration(self.batch_size * self.epochs)
        
        batch_num = len(s_e) // self.batch_size
        index = np.arange(len(s_e))
        np.random.shuffle(index)
        loss = 0
        for i in range(batch_num):
            idx = index[i*self.batch_size : (i+1)*self.batch_size]
            state = s_e[idx]
            action = a_e[idx]

            # update actor
            with tf.GradientTape() as tape:
                pred_action = self.actor(self.encoder, state, sampling=True)   # (N, A)
                # CE
                actor_loss = -tf.reduce_mean(tf.reduce_sum(action * tf.math.log(pred_action + 1e-8), axis=1))
            grads = tape.gradient(actor_loss, self.vars)
            if self.grad_global_norm > 0:
                grads, _ = tf.clip_by_global_norm(grads, self.grad_global_norm)
            self.opt.apply_gradients(zip(grads, self.vars))
            loss += actor_loss.numpy()
        return loss / batch_num
            

    def save_model(self, dir, tag=''):
        self.actor.save_weights(dir + tag + 'actor.h5')
        self.encoder.save_weights(dir + tag + 'encoder.h5')

    def load_model(self, dir, tag=''):
        if os.path.exists(dir + tag + 'actor.h5'):
            self.actor.load_weights(dir + tag + 'actor.h5')
            print('Actor loaded... %s%sactor.h5' % (dir, tag))
        if os.path.exists(dir + tag + 'encoder.h5'):
            self.encoder.load_weights(dir + tag + 'encoder.h5')
            print('encoder loaded... %s%sencoder.h5' % (dir, tag))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--render',     action='store_true')
    parser.add_argument('--verbose',    action='store_true')
    parser.add_argument('--global_norm',type=float, default=0.0)
    parser.add_argument('--actor_lr',   type=float, default=1e-4)
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--batch_size', type=int,   default=128)
    parser.add_argument('--save_rate',  type=int,   default=20)
    parser.add_argument('--env',        type=str,   default='Navi-v1')
    parser.add_argument('--log_name',   type=str,   default='bc')
    parser.add_argument('--actor_units',        type=int,   nargs='*', default=[256, 128])

    args = parser.parse_args()

    env_name = args.env
    data_dir = 'data/%s/' % env_name
    model_dir = 'weights/%s/' % args.log_name
    demo_list = os.listdir(data_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    agent = BC(
        global_norm=args.global_norm,
        actor_lr=args.actor_lr,
        actor_units=args.actor_units,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=data_dir,
        demo_list=demo_list
    )
    
    if args.load_model:
        agent.load_model(model_dir)
        
   
    # Environment Interaction Iteration
    epoch = 0
    stats = []
    while True:
        # reset demo
        if epoch % agent.demo_rotate == 0:
            agent.set_demo()
        loss = agent.update()
        if args.verbose:
            print('E:%d... loss: %.4f\t\t\t' % (epoch, loss), end='\r')
        # done
        stats.append([loss])
        if epoch % args.save_rate == 0:    
            m_loss = np.mean(stats, axis=0)
            agent.save_model(model_dir)
            with open('%s.csv' % args.log_name, 'a', newline='') as f:
                wrt = csv.writer(f)
                for row in stats:
                    wrt.writerow(row)
            stats.clear()
        epoch += 1
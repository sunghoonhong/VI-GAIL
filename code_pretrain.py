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
from models.code_vae import Actor, DiscretePosterior
from models.VAE_Network import VAE_Encoder
from utils.memory import HorizonMemory, ReplayMemory
from utils.util import *
from utils.config import *


class CodeVAE:
    def __init__(self, global_norm, lr, actor_units, code_units, 
                epochs, batch_size, data_dir, demo_list):
        # build network
        self.actor  = Actor(lr=lr, hidden_units=actor_units)
        self.prior  = DiscretePosterior(lr=lr, hidden_units=code_units)
        self.encoder = VAE_Encoder(latent_num=64)
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)

        # set hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_global_norm = global_norm
        self.init_temperature = 2.0
        self.temperature = self.init_temperature
        self.min_temperature = 0.5
        self.temp_decay = 1e-3
        self.beta = 1e-4

        # build expert demonstration Pipeline
        self.data_dir = data_dir
        self.demo_list = os.listdir(data_dir)
        self.demo_group_num = 500
        self.demo_rotate = 3
        assert len(demo_list) >= self.demo_group_num
        self.set_demo()
        self.total = 0
        # ready
        self.dummy_forward()
        self.vars = self.actor.trainable_variables + self.prior.trainable_variables

    def dummy_forward(self):
        # connect networks
        dummy_state = np.zeros([1] + STATE_SHAPE, dtype=np.float32)
        dummy_action = np.zeros([1] + ACTION_SHAPE, dtype=np.float32)
        dummy_code = np.zeros([1] + [DISC_CODE_NUM], dtype=np.float32)
        self.encoder(dummy_state)
        self.prior(self.encoder, dummy_state, dummy_action, dummy_code)
        self.actor(self.encoder, dummy_state, dummy_code)

    def set_demo(self):
        self.demo_list = os.listdir(data_dir)
        selected_demos = random.sample(self.demo_list, self.demo_group_num)

        self.expert_states = []
        self.expert_actions = []
        for demo_name in selected_demos:
            demo = np.load(self.data_dir + demo_name)
            states = demo['state']
            actions = demo['action']
    
            self.expert_states.append(states)
            self.expert_actions.append(actions)
        # self.expert_states = np.concatenate(expert_states, axis=0)
        # self.expert_actions = np.concatenate(expert_actions, axis=0)
        del demo

    def update_temperature(self, epoch):
        self.temperature = \
            max(self.min_temperature, self.init_temperature * math.exp(-self.temp_decay * epoch))

    def label_prev_code(self, s, a):
        # sequential labeling
        prev_codes = []
        running_code = np.eye(DISC_CODE_NUM, dtype=np.float32)[
            [np.random.randint(0, DISC_CODE_NUM)]
        ]   # initial code
        # c_0 ~ c_t-1   [N-1, C]
        for t in range(1, len(s)):
            # s_t a_t-1 c_t-1 -> c_t
            prev_codes.append(running_code)
            running_code = self.prior(self.encoder, s[t:t+1], a[t-1:t], running_code).numpy()
            running_code = np.eye(DISC_CODE_NUM, dtype=np.float32)[
                [np.random.choice(DISC_CODE_NUM, p=running_code[0])]
            ]
        
        return np.concatenate(prev_codes, axis=0)

    def update(self):
        # load expert demonstration
        # states, prev_actions = self.get_demonstration()
        # about 20000 samples
        states = []
        actions = []
        prev_actions = []
        prev_codes = []
        for s, a in zip(self.expert_states, self.expert_actions):
            states.append(s[1:])    # s_1: s_t
            prev_actions.append(a[:-1]) # a_0 : a_t-1
            actions.append(a[1:])   # a_1 : a_t
            prev_code = self.label_prev_code(s, a)  # c_0 : c_t-1
            prev_codes.append(prev_code)
        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        prev_actions = np.concatenate(prev_actions, axis=0)
        prev_codes = np.concatenate(prev_codes, axis=0)
        # print(prev_codes)
        batch_num = len(states) // self.batch_size
        index = np.arange(len(states))
        loss = 0
        for epoch in range(self.epochs):
            np.random.shuffle(index)
            for i in range(batch_num):
                idx = index[i*self.batch_size : (i+1)*self.batch_size]
                state = states[idx]             # (N, S) s_t
                action = actions[idx]           # (N, A) a_t
                prev_action = prev_actions[idx] # (N, A) a_t-1
                prev_code = prev_codes[idx]     # (N, C) c_t-1

                # update vae
                with tf.GradientTape() as tape:
                    code = self.prior(self.encoder, state, prev_action, prev_code)  # (N, C) c_t
                    sampled_code = tf_reparameterize(code, self.temperature)
                    policy = self.actor(self.encoder, state, sampled_code)   # (N, A) a_t
                    log_probs = tf.math.log(sampled_code + 1e-8)
                    log_prior_probs = tf.math.log(1 / DISC_CODE_NUM)
                    kld_loss = tf.reduce_mean(tf.reduce_sum(sampled_code * (log_probs - log_prior_probs), axis=1))
                    actor_loss = -tf.reduce_mean(tf.reduce_sum(action * tf.math.log(policy + 1e-8), axis=1)) # (N-1, )
                    
                    vae_loss = self.beta * kld_loss + actor_loss
                    print(('{:.2f} '*4).format(*policy.numpy()[100]) +' / '+ ('{:.2f} ' * 4).format(*code.numpy()[100])+' / ' +
                            ('{:.2f} '*4).format(*sampled_code.numpy()[100]), '%.2f %.2f %.2f %.2f' %(vae_loss.numpy(), kld_loss.numpy(), actor_loss.numpy(), self.temperature), end='\r')
                grads = tape.gradient(vae_loss, self.vars)
                if self.grad_global_norm > 0:
                    grads, _ = tf.clip_by_global_norm(grads, self.grad_global_norm)
                self.opt.apply_gradients(zip(grads, self.vars))
                loss += vae_loss.numpy()
                self.total += 1
                self.update_temperature(self.total)
                
        loss /= self.epochs * batch_num
        return loss
            

    def save_model(self, dir, tag=''):
        self.actor.save_weights(dir + tag + 'actor.h5')
        self.prior.save_weights(dir + tag + 'prior.h5')

    def load_model(self, dir, tag=''):
        if os.path.exists(dir + tag + 'actor.h5'):
            self.actor.load_weights(dir + tag + 'actor.h5')
            print('Actor loaded... %s%sactor.h5' % (dir, tag))
        if os.path.exists(dir + tag + 'prior.h5'):
            self.prior.load_weights(dir + tag + 'prior.h5')
            print('prior loaded... %s%sprior.h5' % (dir, tag))
    
    def load_encoder(self, dir, tag=''):
        if os.path.exists(dir + tag + 'encoder.h5'):
            self.encoder.load_weights(dir + tag + 'encoder.h5')
            print('Encoder loaded... %s%sencoder.h5' % (dir, tag))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--init_enc',   action='store_true')
    parser.add_argument('--test',       action='store_true')
    parser.add_argument('--render',     action='store_true')
    parser.add_argument('--verbose',    action='store_true')
    parser.add_argument('--global_norm',type=float, default=0.0)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--epochs',     type=int,   default=1)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--save_rate',  type=int,   default=20)
    parser.add_argument('--env',        type=str,   default='Navi-v1')
    parser.add_argument('--log_name',   type=str,   default='code')
    parser.add_argument('--actor_units',type=int,   nargs='*', default=[256, 128])
    parser.add_argument('--code_units', type=int,   nargs='*', default=[64, 64])

    args = parser.parse_args()
    env_name = args.env
    data_dir = 'data/%s/' % env_name
    model_dir = 'weights/%s/' % args.log_name
    bc_dir = 'weights/bc/'
    demo_list = os.listdir(data_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    agent = CodeVAE(
        global_norm=args.global_norm,
        lr=args.lr,
        actor_units=args.actor_units,
        code_units=args.code_units,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=data_dir,
        demo_list=demo_list
    )
    
    if args.load_model:
        agent.load_model(model_dir)
    if not args.init_enc:
        agent.load_encoder(bc_dir)

    epoch = 0
    stats = []
    while True:
        # reset demo
        if epoch % agent.demo_rotate == 0:
            # start = time.time()
            agent.set_demo()
            # print('load demo', time.time()-start)
        # start=time.time()
        loss = agent.update()
        # print('update', time.time() - start)
        if args.verbose:
            print('E:%d... loss: %.4f\t\t\t' % (epoch, loss), end='\r')
        # done
        stats.append([loss, agent.temperature])
        if not args.test and epoch % args.save_rate == 0:    
            m_loss, m_temp = np.mean(stats, axis=0)
            agent.save_model(model_dir)
            with open('%s.csv' % args.log_name, 'a', newline='') as f:
                wrt = csv.writer(f)
                for row in stats:
                    wrt.writerow(row)
            stats.clear()
        epoch += 1
        
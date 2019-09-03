import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='2' # '1', '2', '3', '0,1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import csv
import time
import math
import random
import argparse
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf

# from env.navigate import NaviEnv
from env.navigate_track import NaviEnvTrack
from models.divail import Actor, Critic, Discriminator
from models.code_vae import DiscretePosterior
from models.VAE_Network import VAE_Encoder
from utils.memory import HorizonMemory
from utils.util import *
from utils.config import *

'''
1. Learned expert random episode {'image': (N, 32, 32, 3), 'enc': (N, 64)}
2. Code Frequency per Episode (histogram?)
3. Trajectory plotting
'''


class GAIL:
    def __init__(self, reward_shift, actor_units, critic_units, disc_units, disc_reduce_units, code_units):
        # build network
        self.actor  = Actor(lr=0, hidden_units=actor_units)
        self.critic = Critic(lr=0, hidden_units=critic_units)
        self.discriminator = Discriminator(
            lr=0, hidden_units=disc_units, reduce_units=disc_reduce_units)
        self.encoder = VAE_Encoder(latent_num=64)
        self.prior = DiscretePosterior(lr=0, hidden_units=code_units)

        # set hyperparameters
        self.reward_shift = reward_shift
        self.memory = HorizonMemory()

        # ready
        self.dummy_forward()

    def dummy_forward(self):
        # connect networks
        dummy_state = np.zeros([1] + STATE_SHAPE, dtype=np.float32)
        dummy_action = np.zeros([1] + ACTION_SHAPE, dtype=np.float32)
        dummy_code = np.zeros([1, DISC_CODE_NUM], dtype=np.float32)
        self.encoder(dummy_state)
        self.prior(self.encoder, dummy_state, dummy_action, dummy_code)
        self.actor(self.encoder, dummy_state, dummy_code)
        self.critic(self.encoder, dummy_state)
        self.discriminator(self.encoder, dummy_state, dummy_action)

    def get_code(self, state, prev_action, prev_code):
        code_prob = self.prior(self.encoder, state, prev_action, prev_code).numpy()[0]
        code_idx = np.argmax(code_prob) # greedy
        # code_idx = np.random.choice(DISC_CODE_NUM, p=code_prob)
        code = np.eye(DISC_CODE_NUM, dtype=np.float32)[[code_idx]]  # (1, C)
        return code_idx, code, code_prob

    def get_action(self, state, code):
        policy = self.actor(self.encoder, state, code).numpy()[0]
        # action = np.random.choice(ACTION_NUM, p=policy)
        action = np.argmax(policy)   # greedy
        action_one_hot = np.eye(ACTION_NUM, dtype=np.float32)[[action]] # (1, 4)
        log_old_pi = [[np.log(policy[action] + 1e-8)]]  # (1, 1)
        return action, action_one_hot, log_old_pi, policy

    def get_reward(self, states, actions):
        d = self.discriminator(self.encoder, states, actions).numpy()   # (N, 1)
        # rewards = 0.5 - d       # linear reward 
        # rewards = np.tan(0.5 - d)     # tan reward
        if self.reward_shift:
            rewards = -np.log(2.0 * d + 1e-8)    # log equil reward   
        else:
            rewards = -np.log(d + 1e-8)   # log reward
        # rewards = 0.1 * np.where(rewards>1, 1, rewards)
        return rewards
    
    def load_model(self, dir, tag=''):
        if os.path.exists(dir + tag + 'actor.h5'):
            self.actor.load_weights(dir + tag + 'actor.h5')
            print('Actor loaded... %s%sactor.h5' % (dir, tag))
        if os.path.exists(dir + tag + 'critic.h5'):
            self.critic.load_weights(dir + tag + 'critic.h5')
            print('Critic loaded... %s%scritic.h5' % (dir, tag))
        if os.path.exists(dir + tag + 'discriminator.h5'):
            self.discriminator.load_weights(dir + tag + 'discriminator.h5')
            print('Discriminator loaded... %s%sdiscriminator.h5' % (dir, tag))
    
    def load_encoder(self, dir, tag=''):
        if os.path.exists(dir + tag + 'encoder.h5'):
            self.encoder.load_weights(dir + tag + 'encoder.h5')
            print('encoder loaded... %s%sencoder.h5' % (dir, tag))

    def load_prior(self, dir, tag=''):
        if os.path.exists(dir + tag + 'prior.h5'):
            self.prior.load_weights(dir + tag + 'prior.h5')
            print('prior loaded... %s%sprior.h5' % (dir, tag))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_vae',   action='store_true')
    parser.add_argument('--unload_bc',  action='store_true')
    parser.add_argument('--unload_prior',action='store_true')
    parser.add_argument('--load_bc_actor',action='store_true')
    parser.add_argument('--render',     action='store_true')
    parser.add_argument('--verbose',    action='store_true')
    parser.add_argument('--reward_shift', action='store_true')
    parser.add_argument('--env',        type=str,   default='Navi-v1')
    parser.add_argument('--log_name',   type=str,   default='digail_GE_LS_V')
    parser.add_argument('--delay',      type=float, default=0.)
    parser.add_argument('--episodes',   type=int,   default=1)
    parser.add_argument('--actor_units',        type=int,   nargs='*', default=[256, 128])
    parser.add_argument('--critic_units',       type=int,   nargs='*', default=[256, 128])
    parser.add_argument('--disc_units',         type=int,   nargs='*', default=[256])
    parser.add_argument('--disc_reduce_units',  type=int,   nargs='*', default=[])
    parser.add_argument('--code_units', type=int,   nargs='*', default=[64, 64])

    args = parser.parse_args()
    args.load_model = True
    args.verbose=True
    env_name = args.env
    model_dir = 'weights/%s/' % args.log_name
    vae_dir = 'weights/vaecnn/'
    bc_dir = 'weights/bc/'
    prior_dir = 'weights/code/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    agent = GAIL(
        reward_shift=args.reward_shift,
        actor_units=args.actor_units,
        critic_units=args.critic_units,
        disc_units=args.disc_units,
        disc_reduce_units=args.disc_reduce_units,
        code_units=args.code_units
    )
    
    if args.load_model:
        agent.load_model(model_dir)
    if args.load_vae:
        agent.load_encoder(vae_dir)
    if not args.unload_bc:
        agent.load_encoder(bc_dir)
    if not args.unload_prior:
        agent.load_prior(prior_dir)
    if args.load_bc_actor:
        agent.load_model(bc_dir)
        
    # prepare Environment and Demonstration
    env = NaviEnvTrack()

    best_score = 0
    stats = []
    # Environment Interaction Iteration
    for episode in range(args.episodes):
        # Reset
        code_probs = []
        code_freq = np.zeros((DISC_CODE_NUM), dtype=np.float32)
        done = False
        score = 0
        disc_rewards = 0
        timestep = 0

        pmax = 0

        obs = env.reset()
        state = preprocess_obs(obs)
        # random initial prev action & code
        prev_action = np.eye(ACTION_NUM, dtype=np.float32)[
            [np.random.randint(0, ACTION_NUM)]
        ]   # (1, A)
        prev_code = np.eye(DISC_CODE_NUM, dtype=np.float32)[
            [np.random.randint(0, DISC_CODE_NUM)]
        ]   # (1, C)
        while not done and timestep < 100:
            timestep += 1
            if args.render:
                time.sleep(args.delay)
                env.render()

            real_code, code, code_prob = agent.get_code(state, prev_action, prev_code)

            code_freq[real_code] += 1.  # for code frequency stat
            code_probs.append([code_prob])
            real_action, action, log_old_pi, policy \
                            = agent.get_action(state, code)
          
            disc_reward = float(agent.get_reward(state, action).reshape(-1))
            obs, reward, done, info = env.step(real_action, real_code)
            
            if args.verbose:
                step_temp = '[E{:d} T{:d}] '.format(episode, timestep)
                action_temp = 'Act: %d (%.3f) ' % (real_action, policy[real_action])
                pi_temp = ('pi:[' + ' {:.2f}' * ACTION_NUM + '] ').format(*policy)
                option_temp = 'Opt: %d (%.3f) ' % (real_code, code_prob[real_code])
                code_temp = ('code:[' + ' {:.2f}' * DISC_CODE_NUM + '] ').format(*code_prob)
                print(step_temp + action_temp + pi_temp + option_temp + code_temp, end='\r')
            
            next_state = preprocess_obs(obs)

            agent.memory.append(state, action, reward, log_old_pi, code)

            state = next_state
            prev_action = action
            prev_code = code
            score += reward
            disc_rewards += disc_reward
            pmax += np.amax(policy)

        code_freq /= timestep
        if args.verbose:
            print('Ep %d: Score %d Reward %.3f Step %d\t\t\t' % (episode, score, disc_rewards, timestep))
            print('Code freq:', ('{:.2f} ' * 4).format(*code_freq))
        # done
        agent.memory.flush()
        plt.figure(figsize=(10,7))
        ax = plt.gca()
        ax.xaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_tick_params(labelsize=30)
        plt.xticks(list(range(DISC_CODE_NUM)))
        # plot code freq histogram
        # plt.figure(figsize=(10,6))
        x = np.arange(DISC_CODE_NUM)
        y = code_freq
        plt.xticks(list(range(DISC_CODE_NUM)))
        plt.bar(x,y, color=['tab:pink', '#FFF000', (0., 0.35, 1.), 'r'])
        #plt.title('Code Frequency', position=(0.5, -0.15))
        plt.savefig('code_freq.png')
        plt.clf()
        ax = plt.gca()
        ax.xaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_tick_params(labelsize=30)
        # plot code probs history graph
        x = np.arange(len(code_probs))
        code_probs = np.concatenate(code_probs, axis=0)
        y1 = code_probs[:, 0]
        y2 = code_probs[:, 1]
        y3 = code_probs[:, 2]
        y4 = code_probs[:, 3]
        #plt.title('Code Distribution', position=(0.5, -0.15))
        plt.plot(x, y1, 'tab:pink', label=0)
        plt.plot(x, y2, '#FFF000', label=1)
        plt.plot(x, y3, 'b', label=2)
        plt.plot(x, y4, 'r', label=3)
                
        plt.legend(loc='lower left', fontsize=25)
        plt.savefig('code_dist.png')
        plt.clf()



        tr_img = env.track_render()
        # tr_img = cv2.cvtColor(tr_img, cv2.COLOR_BGR2RGB)
        # tr_img = Image.fromarray(tr_img[..., [2,1,0]]).rotate(180)
        tr_img = ImageOps.mirror(Image.fromarray(tr_img).rotate(270))
        tr_img.save('track.png')
        # stats.append([disc_rewards, score, timestep, pmax/timestep])
        
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
# import gym

import cv2
import tensorflow as tf

from env.navigate import NaviEnv
from models.digail import Actor, Critic, Discriminator
from models.code_vae import DiscretePosterior
from models.VAE_Network import VAE_Encoder
from utils.memory import HorizonMemory, ReplayMemory
from utils.util import *
from utils.config import *


class GAIL:
    def __init__(self, reward_shift, reward_aug, gae_norm, global_norm, actor_lr, critic_lr, disc_lr,
                actor_units, critic_units, disc_units, disc_reduce_units, code_units,
                gamma, lambd, clip, entropy, info_weight, epochs, batch_size, update_rate,
                data_dir, demo_list):
        # build network
        self.actor  = Actor(lr=actor_lr, hidden_units=actor_units)
        self.critic = Critic(lr=critic_lr, hidden_units=critic_units)
        self.discriminator = Discriminator(
            lr=disc_lr, hidden_units=disc_units, reduce_units=disc_reduce_units)
        self.encoder = VAE_Encoder(latent_num=64)
        self.prior = DiscretePosterior(lr=0, hidden_units=code_units)

        # set hyperparameters
        self.reward_shift = reward_shift
        self.reward_aug = reward_aug
        self.gae_norm = gae_norm
        self.gamma = gamma
        self.lambd = lambd
        self.gam_lam = gamma * lambd
        self.clip = clip
        self.entropy = entropy
        self.info_weight = info_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.half_batch_size = batch_size // 2
        self.update_rate = update_rate
        self.grad_global_norm = global_norm

        # build memory
        self.memory = HorizonMemory(use_reward=reward_aug, use_code=True)
        self.replay = ReplayMemory(use_code=True)

        # build expert demonstration Pipeline
        self.data_dir = data_dir
        self.demo_list = os.listdir(data_dir)
        self.demo_group_num = 500
        self.demo_rotate = 5
        assert len(demo_list) >= self.demo_group_num
        self.set_demo()

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
        
    def memory_process(self, next_state, done, next_code):
        if self.reward_aug:
            states, actions, log_old_pis, codes, rewards = self.memory.rollout()
        else:
            states, actions, log_old_pis, codes = self.memory.rollout()
        np_states = np.concatenate(states + [next_state], axis=0)
        np_actions = np.concatenate(actions, axis=0)

        next_states = states[1:] + [next_state]
        next_codes = codes[1:] + [next_code]

        np_rewards = self.get_reward(np_states[:-1], np_actions)  # (N, 1)
        if self.reward_aug:
            np_env_rewards = np.stack(rewards, axis=0).reshape(-1, 1)
            np_rewards = np_rewards + np_env_rewards
        gae, oracle = self.get_gae_oracle(np_states, np_rewards, done) # (N, 1), (N, 1)
        self.replay.append(states, actions, log_old_pis, gae, oracle, next_states, codes, next_codes)
        self.memory.flush()
        if len(self.replay) >= self.update_rate:
            self.update()
            self.replay.flush()

    def get_code(self, state, prev_action, prev_code):
        code_prob = self.prior(self.encoder, state, prev_action, prev_code).numpy()[0]
        code_idx = np.random.choice(DISC_CODE_NUM, p=code_prob)
        code = np.eye(DISC_CODE_NUM, dtype=np.float32)[[code_idx]]  # (1, C)
        return code_idx, code, code_prob

    def get_action(self, state, code):
        policy = self.actor(self.encoder, state, code).numpy()[0]
        action = np.random.choice(ACTION_NUM, p=policy)
        # action = np.argmax(policy)
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
    
    def get_gae_oracle(self, states, rewards, done):
        # states include next state
        values = self.critic(self.encoder, states).numpy()   # (N+1, 1)
        if done:
            values[-1] = np.float32([0])
        N = len(rewards)
        gae = 0
        gaes = np.zeros((N, 1), dtype=np.float32)
        oracles = np.zeros((N, 1), dtype=np.float32)
        for t in reversed(range(N)):
            oracles[t] = rewards[t] + self.gamma * values[t+1]
            delta = oracles[t] - values[t]
            gae = delta + self.gam_lam * gae
            gaes[t][0] = gae
        
        # oracles = gaes + values[:-1]        # (N, 1)
        if self.gae_norm:
            gaes = (gaes - np.mean(gaes)) / (np.std(gaes) + 1e-8)
        return gaes, oracles

    def update(self):
        # load & calculate data
        states, actions, log_old_pis, gaes, oracles, next_states, codes, next_codes \
            = self.replay.rollout()

        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        log_old_pis = np.concatenate(log_old_pis, axis=0)
        gaes = np.concatenate(gaes, axis=0)
        oracles = np.concatenate(oracles, axis=0)
        next_states = np.concatenate(next_states, axis=0)
        codes = np.concatenate(codes, axis=0)
        next_codes = np.concatenate(next_codes, axis=0)

        N = len(states)
        # update discriminator
        # load expert demonstration
        s_e, a_e = self.get_demonstration(N)
        
        batch_num = N // self.half_batch_size
        index = np.arange(N)
        np.random.shuffle(index)
        for i in range(batch_num):
            idx = index[i*self.half_batch_size : (i+1)*self.half_batch_size]
            s_concat = np.concatenate([states[idx], s_e[idx]], axis=0)
            a_concat = np.concatenate([actions[idx], a_e[idx]], axis=0)
            
            with tf.GradientTape() as tape:
                discs = self.discriminator(self.encoder, s_concat, a_concat)
                agent_loss = -tf.reduce_mean(tf.math.log(discs[:self.half_batch_size] + 1e-8))
                expert_loss = -tf.reduce_mean(tf.math.log(1 + 1e-8 - discs[self.half_batch_size:]))
                disc_loss = agent_loss + expert_loss
            disc_grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)
            if self.grad_global_norm > 0:
                disc_grads, _ = tf.clip_by_global_norm(disc_grads, self.grad_global_norm)
            self.discriminator.opt.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        # TODO: update posterior
        # L1 loss = logQ(code|s,prev_a,prev_code)
        # update actor & critic
        # batch_num = math.ceil(len(states) / self.batch_size)
        batch_num = N // self.batch_size
        index = np.arange(N)
        for _ in range(self.epochs):
            np.random.shuffle(index)
            for i in range(batch_num):
                # if i == batch_num - 1:
                #     idx = index[i*self.batch_size : ] 
                # else:
                idx = index[i*self.batch_size : (i+1)*self.batch_size]
                state = states[idx]
                action = actions[idx]
                log_old_pi = log_old_pis[idx]
                gae = gaes[idx]
                oracle = oracles[idx]
                next_state = next_states[idx]
                code = codes[idx]
                next_code = next_codes[idx]
                
                # update critic
                with tf.GradientTape() as tape:
                    values = self.critic(self.encoder, state)     # (N, 1)
                    critic_loss = tf.reduce_mean((oracle - values) ** 2)   # MSE loss
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                if self.grad_global_norm > 0:
                    critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.grad_global_norm)
                self.critic.opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

                # update actor
                with tf.GradientTape() as tape:
                    pred_action = self.actor(self.encoder, state, code)     # s_t, c_t -> a_t
                    pred_next_code = self.prior(self.encoder, next_state, pred_action, code) # s_t+1, a_t, c_t -> c_t+1
                    # Directed Information term
                    info_loss = -tf.reduce_mean(tf.reduce_sum(next_code * tf.math.log(pred_next_code + 1e-8), axis=1))

                    # RL (PPO) term
                    log_pi = tf.expand_dims(tf.math.log(tf.reduce_sum(pred_action * action, axis=1) + 1e-8), axis=1)    # (N, 1)
                    ratio = tf.exp(log_pi - log_old_pi)
                    clip_ratio = tf.clip_by_value(ratio, 1 - self.clip, 1 + self.clip)
                    clip_loss = -tf.reduce_mean(tf.minimum(ratio * gae, clip_ratio * gae))
                    entropy = tf.reduce_mean(tf.exp(log_pi) * log_pi)
                    actor_loss = clip_loss + self.entropy * entropy + self.info_weight * info_loss

                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)       # NOTE: freeze posterior
                if self.grad_global_norm > 0:
                    actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.grad_global_norm)
                self.actor.opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            print('%d samples trained... D loss: %.4f C loss: %.4f A loss: %.4f\t\t\t'
                % (len(gaes), disc_loss, critic_loss, actor_loss), end='\r')
            

    def save_model(self, dir, tag=''):
        self.actor.save_weights(dir + tag + 'actor.h5')
        self.critic.save_weights(dir + tag + 'critic.h5')
        self.discriminator.save_weights(dir + tag + 'discriminator.h5')

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
    parser.add_argument('--gae_norm',   action='store_true')
    parser.add_argument('--reward_aug', action='store_true')
    parser.add_argument('--reward_shift', action='store_true')
    parser.add_argument('--global_norm',type=float, default=0.0)
    parser.add_argument('--actor_lr',   type=float, default=1e-4)
    parser.add_argument('--critic_lr',  type=float, default=2.5e-4)
    parser.add_argument('--disc_lr',    type=float, default=2.5e-5)
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--lambd',      type=float, default=0.98)
    parser.add_argument('--clip',       type=float, default=0.15)
    parser.add_argument('--entropy',    type=float, default=1e-3)
    parser.add_argument('--info_weight',    type=float, default=1e-1)
    parser.add_argument('--epochs',     type=int,   default=5)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--horizon',    type=int,   default=8)
    parser.add_argument('--update_rate',type=int,   default=512)
    parser.add_argument('--save_rate',  type=int,   default=20)
    parser.add_argument('--env',        type=str,   default='Navi-v1')
    parser.add_argument('--log_name',   type=str,   default='gail')
    parser.add_argument('--actor_units',        type=int,   nargs='*', default=[256, 128])
    parser.add_argument('--critic_units',       type=int,   nargs='*', default=[256, 128])
    parser.add_argument('--disc_units',         type=int,   nargs='*', default=[256])
    parser.add_argument('--disc_reduce_units',  type=int,   nargs='*', default=[])
    parser.add_argument('--code_units', type=int,   nargs='*', default=[64, 64])

    args = parser.parse_args()

    env_name = args.env
    data_dir = 'data/%s/' % env_name
    model_dir = 'weights/%s/' % args.log_name
    vae_dir = 'weights/vaecnn/'
    bc_dir = 'weights/bc/'
    prior_dir = 'weights/code/'
    demo_list = os.listdir(data_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    agent = GAIL(
        reward_shift=args.reward_shift,
        reward_aug=args.reward_aug,
        gae_norm=args.gae_norm,
        global_norm=args.global_norm,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        disc_lr=args.disc_lr,
        actor_units=args.actor_units,
        critic_units=args.critic_units,
        disc_units=args.disc_units,
        disc_reduce_units=args.disc_reduce_units,
        code_units=args.code_units,
        gamma=args.gamma,
        lambd=args.lambd,
        clip=args.clip,
        entropy=args.entropy,
        info_weight=args.info_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        update_rate=args.update_rate,
        data_dir=data_dir,
        demo_list=demo_list
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
    env = NaviEnv()

    global_step = 0
    episode = 0
    best_score = 0
    stats = []
    # Environment Interaction Iteration
    while True:
        # Reset
        episode += 1
        done = False
        score = 0
        disc_rewards = 0
        timestep = 0
        horizon_step = 0
        pmax = 0
        global_step += 1

        # reset demo
        if episode % agent.demo_rotate == 0:
            agent.set_demo()

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
            global_step += 1
            timestep += 1
            horizon_step += 1
            if args.render:
                env.render()

            real_code, code, code_prob = agent.get_code(state, prev_action, prev_code)
            real_action, action, log_old_pi, policy \
                            = agent.get_action(state, code)
          
            disc_reward = float(agent.get_reward(state, action).reshape(-1))

            obs, reward, done, info = env.step(real_action)
            
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

            if horizon_step == args.horizon or done:
                horizon_step = 0
                _, next_code, _ = agent.get_code(next_state, action, code)
                agent.memory_process(next_state, done, next_code)
        if args.verbose:
            print('Ep %d: Score %d Reward %.3f Step %d\t\t\t' % (episode, score, disc_rewards, timestep))

        # done
        stats.append([disc_rewards, score, timestep, pmax/timestep])
        if episode % args.save_rate == 0:    
            m_disc_reward, m_score, m_step, m_pmax = np.mean(stats, axis=0)
            if m_score > best_score:
                best_score = m_score
                agent.save_model(model_dir, 'best')
            agent.save_model(model_dir)
            if args.verbose:
                print('E%d: reward:%.2f score:%d step:%d pmax:%.4f\t\t\t'
                    % (episode, m_disc_reward, m_score, m_step, m_pmax))
            with open('%s.csv' % args.log_name, 'a', newline='') as f:
                wrt = csv.writer(f)
                for row in stats:
                    wrt.writerow(row)
            stats.clear()
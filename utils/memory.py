import numpy as np
from utils.config import *


'''
    Memory for Agent

''' 

class HorizonMemory:
    '''
        Short-Term Memory for Multi-Step GAE
    '''
    def __init__(self, use_code=False, use_reward=False):
        '''            
                States: [(84, 84, 4), ...]
                Action: [(4,), ...]
                Reward: [(1,), ...]
                log old pi: [(1,), ...]
                Code:   [(1, C), ...]
        '''
        self.use_code = use_code
        self.use_reward = use_reward
        self.flush()
        
    def __len__(self):
        return len(self.states)

    def flush(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_old_pis = []
        if self.use_code:
            self.codes = []

    def append(self, state, action, reward, log_old_pi, code=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_old_pis.append(log_old_pi)
        if self.use_code:
            self.codes.append(code)

    def rollout(self):
        '''
        return list of states, list of actions, list of log old pi
        '''
        if self.use_code and self.use_reward:
            return self.states, self.actions, self.log_old_pis, self.codes, self.rewards
        elif self.use_code:
            return self.states, self.actions, self.log_old_pis, self.codes
        elif self.use_reward:
            return self.states, self.actions, self.log_old_pis, self.rewards
        else:
            return self.states, self.actions, self.log_old_pis



class ReplayMemory:
    '''
        Long-Term Mermoy for Update PPO Agent
    '''
    def __init__(self, use_code=False):
        self.use_code = use_code
        self.flush()

    def __len__(self):
        return len(self.states)

    def flush(self):
        self.states = []
        self.actions = []
        self.log_old_pis = []
        self.gaes = []
        self.oracles = []
        if self.use_code:
            self.next_states = []
            self.codes = []
            self.next_codes = []

    def append(self, state, action, log_old_pi, gae, oracle, next_state=None, code=None, next_code=None):
        self.states.extend(state)
        self.actions.extend(action)
        self.log_old_pis.extend(log_old_pi)
        self.gaes.extend(gae)
        self.oracles.extend(oracle)
        if self.use_code:
            self.next_states.extend(next_state)
            self.codes.extend(code)
            self.next_codes.extend(next_code)

    def rollout(self):
        if self.use_code:
            return self.states, self.actions, self.log_old_pis, self.gaes, self.oracles, \
                self.next_states, self.codes, self.next_codes
        else:
            return self.states, self.actions, self.log_old_pis, self.gaes, self.oracles

    
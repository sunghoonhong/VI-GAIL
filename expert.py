import os
import time
import random
import cv2
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from env.navigate import NaviEnv
from utils.util import *
from utils.memory import HorizonMemory

class Expert:
    def __init__(self):
        self.memory = HorizonMemory()
        
    def solve(self, game):
        # dijkstra
        start = np.array(game.init_pos)
        key = np.array(game.init_key_pos)
        car = np.array(game.init_car_pos)
        graph = np.array(game.map)[1:-1, 1:-1]
        key_path = self.dijkstra(start, key, graph, 0)
        car_path = self.dijkstra(key, car, graph, 1)
        self.actions = key_path + car_path
    
    def dijkstra(self, start, end, graph, has_key):
        # return action sequence
        visited = np.zeros_like(graph)
        path = -1 * np.ones_like(graph)
        dist = np.ones_like(graph) * 500

        dist[start[1], start[0]] = 0
        
        while True:
            temp_dist = 500
            x, y = -1, -1
            for i in range(len(visited)):
                for j in range(len(visited[0])):
                    if not visited[i, j] and temp_dist > dist[i, j]:
                        temp_dist = dist[i, j]
                        x, y = i, j
            if temp_dist == 500:
                break

            visited[x, y] = 1
            
            for act, adj in enumerate([(x-1, y), (x+1, y), (x, y-1), (x, y+1)]):
                i, j = adj
                if i in range(0, 7) and j in range(0, 7):
                    if not visited[i, j] and graph[i, j] != 1:
                        if not has_key and graph[i, j] == 3:
                            continue
                        via = dist[x, y] + 1
                        if dist[i, j] > via:
                            dist[i, j] = via
                            path[i, j] = act
        backtrack = []
        temp = np.array([end[1], end[0]])
        
        dest = np.array([start[1], start[0]])
        DIRECTION = [
            np.array([1, 0]),
            np.array([-1, 0]),
            np.array([0, 1]),
            np.array([0, -1])
        ]
        while not np.all(temp == dest):
            act = path[temp[0], temp[1]]
            backtrack.append(act)
            temp += DIRECTION[act]
        return list(reversed(backtrack))
            
    def get_action(self, t):
        act_idx = self.actions[t]
        act = np.eye(4, dtype=np.float32)[[act_idx]]
        return act_idx, act

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--delay',  type=float, default=0)
    parser.add_argument('--episode',type=int, default=1)
    parser.add_argument('--limit',  type=int, default=100)
    args = parser.parse_args()

    data_dir = 'data/Navi-v1/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    env = NaviEnv()
    expert = Expert()

    for e in range(args.episode):
        done = False
        score = 0
        obs = env.reset()

        expert.solve(env.game)
        state = preprocess_obs(obs)
        for t in range(args.limit):
            time.sleep(args.delay)
            if args.render:
                env.render()
            
            real_action, action = expert.get_action(t)
            obs, rew, done, info = env.step(real_action)
            
            next_state = preprocess_obs(obs)
            expert.memory.append(state, action, 0, 0)
            score += rew
            state = next_state
            if done:
                break
        if args.verbose:
            print('E%dT%d: Score %d Optim %d' % (e, t, score, env.game.optim_dist), done)
        
        if done and args.record:
            print('Ep%d' % e, end='\r')
            states, actions, _ = expert.memory.rollout()

            states = np.concatenate(states, axis=0)
            actions = np.concatenate(actions, axis=0)
            demo_id = '%d_%d_%d' % (score, t, e) + datetime.now().strftime('%m_%d_%H_%M_%S')
            filename = data_dir + demo_id
            while os.path.exists(filename + '.npz'):
                filename += '_'
            try:
                np.savez(filename, state=states, action=actions)
            except Exception as e:
                print(str(e))
                if os.path.exists(filename + '.npz'):
                    os.remove(filename + '.npz')
            expert.memory.flush()

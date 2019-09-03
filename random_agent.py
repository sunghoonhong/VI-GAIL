import os
import time
import numpy as np
import cv2
from argparse import ArgumentParser
from env.navigate import NaviEnv
from utils.util import *


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--delay',  type=float, default=0.02)
    parser.add_argument('--episode',type=int, default=2)
    parser.add_argument('--limit',  type=int, default=40)
    args = parser.parse_args()

    env = NaviEnv()
    for e in range(args.episode):
        obs = env.reset()
        if args.record:
            cv2.imwrite('temp.png', preprocess_obs(obs)[0] * 255.)
        done = False
        score = 0
        for t in range(args.limit):
            time.sleep(args.delay)
            if args.render:
                env.render()
            obs, rew, done, info = env.step(np.random.choice(4))
            if args.record:
                cv2.imwrite('temp.png', preprocess_obs(obs)[0] * 255.)
            score += rew
            print('E%dT%d: %d, [%d, %d]' % (e, t, rew, info['pos'][0], info['pos'][1]), end='\r')
            if done:
                break
        if done:
            print('E%dT%d: %d Success' % (e, t, score))
        else:
            print('E%dT%d: %d Fail' % (e, t, score))
        
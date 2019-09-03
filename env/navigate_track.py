import os
import time
import random
import numpy as np
import cv2
import pygame as pg
from pygame import gfxdraw as gdraw
from env.navigate import NaviEnv


G = GRID_LEN = 50
M = 5  # margin
GRID_SIZE = [GRID_LEN] * 2

# color
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE  = (0, 0, 255)
RED100   = (255, 100, 100)
GREEN100 = (100, 255, 100)
BLUE100  = (100, 100, 255)
YELLOW = (255, 255, 0)
SKYBLUE  = (0, 100, 255)
PINK = (255, 0, 255)
BG_COLOR = BLACK

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
DIRECTION = [
    np.array([0, -1]),
    np.array([0, 1]),
    np.array([-1, 0]),
    np.array([1, 0])
]


class NaviEnvTrack(NaviEnv):
    def __init__(self):
        super().__init__()
    
    def reset(self, init_pos=None, key_pos=None, car_pos=None):
        self.game.reset(init_pos=init_pos, key_pos=key_pos, car_pos=car_pos)
        self.game.draw()
        self.track = []
        observe = pg.surfarray.array3d(self.game.screen)
        return observe
        
    def step(self, action, code):
        if self.game.display:
            pg.event.pump()
        self.track.append((self.game.player.rect.topleft, action, code))
        self.game.keydown(action)
        reward = self.game.update()
        done = (not self.game.car)
        self.game.draw()
        observe = pg.surfarray.array3d(self.game.screen)
        info = {'pos': self.game.player.rect.topleft}
        return observe, reward, done, info

    def track_render(self):
        self.game.screen.fill(BG_COLOR)
        for wall in self.game.walls:
            pg.draw.rect(self.game.screen, WHITE, list(wall) + [G] * 2)
        pg.draw.rect(self.game.screen, BLUE100, list(self.game.init_key_pos * G + G // 4) + [G // 2] * 2)
        pg.draw.rect(self.game.screen, RED100, list(self.game.init_car_pos * G + G // 4) + [G // 2] * 2)
        pg.draw.rect(self.game.screen, GREEN100, list(self.game.init_pos * G + G // 4) + [G // 2] * 2)
        for p, a, c in self.track:
            color = [PINK, YELLOW, SKYBLUE, RED][c]
            pos_map = [
                [[p[0]+G//2, p[1]], [p[0]+G//4, p[1]+G//4-M], [p[0]+3*G//4, p[1]+G//4-M]],
                [[p[0]+G//2, p[1]+G], [p[0]+3*G//4, p[1]+3*G//4+M], [p[0]+G//4, p[1]+3*G//4+M]],
                [[p[0], p[1]+G//2], [p[0]+G//4-M, p[1]+3*G//4], [p[0]+G//4-M, p[1]+G//4]],
                [[p[0]+G, p[1]+G//2], [p[0]+3*G//4+M, p[1]+G//4], [p[0]+3*G//4+M, p[1]+3*G//4]]
            ]
            pos = pos_map[a]
            pg.draw.polygon(self.game.screen, color, pos)
        image = pg.surfarray.array3d(self.game.screen)
        return image

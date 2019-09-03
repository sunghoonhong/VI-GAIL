import os
import time
import random
import numpy as np
import pygame as pg
from pygame import gfxdraw as gdraw


WINDOW_HEIGHT, WINDOW_WIDTH = 350, 350
WINDOW_SIZE = [350, 350]
GRID_LEN = 50
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


class Player(pg.sprite.Sprite):
    def __init__(self, pos):
        pg.sprite.Sprite.__init__(self)
        self.rect = pg.Rect([0, 0], GRID_SIZE)
        self.radius = GRID_LEN // 2
        self.direc = np.array([1, 0])
        # self.image = pg.Surface(GRID_SIZE, pg.SRCALPHA)
        self.image = pg.Surface(GRID_SIZE)
        self.image.fill(GREEN100)
        # gdraw.aacircle(self.image, self.rect.center[0], self.rect.center[1], int(self.radius), GREEN100)
        # gdraw.filled_circle(self.image, self.rect.center[0], self.rect.center[1], int(self.radius), GREEN100)
        self.rect.topleft = pos * GRID_LEN
        self.prev_pos = self.rect.topleft
        
        self.has_key = False

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def set_direction(self, direc):
        self.direc = DIRECTION[direc]
    
    def move(self):
        self.prev_pos = self.rect.topleft
        self.rect.center += self.direc * GRID_LEN
    
    def rollback(self):
        self.rect.topleft = self.prev_pos

    def update(self):
        self.move()
        

class Key(pg.sprite.Sprite):
    def __init__(self, pos):
        pg.sprite.Sprite.__init__(self)
        self.pos = pos * GRID_LEN
        self.rect = pg.Rect(self.pos, GRID_SIZE)
        self.image = pg.Surface(GRID_SIZE)
        self.image.fill(BLUE100)
        
        
    def draw(self, screen):
        screen.blit(self.image, self.rect)
        # pos = np.array(self.rect.topleft)
        # half = GRID_LEN // 2
        # top     = [pos[0] + half, pos[1]]
        # left    = [pos[0], pos[1] + GRID_LEN]
        # right   = [pos[0] + GRID_LEN, pos[1] + GRID_LEN]
        # pg.draw.polygon(screen, BLUE100, [top, left, right])


class Car(pg.sprite.Sprite):
    def __init__(self, pos):
        pg.sprite.Sprite.__init__(self)
        self.pos = pos * GRID_LEN
        self.rect = pg.Rect(self.pos, GRID_SIZE)
        self.image = pg.Surface(GRID_SIZE)
        self.image.fill(RED100)

    def draw(self, screen):
        # pos = np.array(self.rect.topleft)
        screen.blit(self.image, self.rect)
        # half = GRID_LEN // 2
        # top     = [pos[0] + half, pos[1]]
        # left    = [pos[0], pos[1] + half]
        # right   = [pos[0] + GRID_LEN, pos[1] + half]
        # down    = [pos[0] + half, pos[1] + GRID_LEN]
        # pg.draw.rect(screen, RED100, [pos[0], pos[1], GRID_LEN, GRID_LEN])
        # pg.draw.polygon(screen, RED, [right, top, left, down])
          
        
class Game:
    def __init__(self):
        '''
        7 * 7 grid world
        K o o x o o o
        o o o o o P o
        o o o x o o o
        x o x x x o x
        o o o x o o o
        o o o o o C o
        o o o x o o o

        P: Player.  Randomly spawned
        K: Key.     First goal
        C: Car.     Second goal. Can be reached only with Key
        '''
        
        self.walls = np.array([
            [0, 3],
            [2, 3],
            [3, 3],
            [4, 3],
            [6, 3],
            [3, 0],
            [3, 2],
            [3, 4],
            [3, 6]
        ]) * GRID_LEN
        
        self.screen = pg.Surface(WINDOW_SIZE)
        self.display = False
        self.reset()
        
    def reset(self):
        self.map = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]
        self.objects = [
            [0, 3], [2, 3], [3, 3], [4, 3], [6, 3], [3, 0],
            [3, 2], [3, 4], [3, 6]
        ]
        self.init_key_pos = np.random.randint(low=0, high=3, size=2)
        self.init_car_pos = np.random.randint(low=4, high=7, size=2)
        self.key = Key(self.init_key_pos)
        self.car = Car(self.init_car_pos)
        self.map[self.init_key_pos[0]+1][self.init_key_pos[1]+1] = 2
        self.map[self.init_car_pos[0]+1][self.init_car_pos[1]+1] = 3
        self.objects += [list(self.init_key_pos)] + [list(self.init_car_pos)]
        while True:
            x = np.random.choice(7)
            y = np.random.choice(7)
            if [x, y] not in self.objects:
                break
        self.init_pos = np.array([x, y])
        self.map[x+1][y+1] = 4
        self.player = Player(self.init_pos)
        self.score = 0
        
    def update(self):
        reward = -1
        self.player.update()
        pos = np.array(self.player.rect.topleft) // GRID_LEN
        status = self.map[pos[0] + 1][pos[1] + 1]
        if status == 1:   # WALL
            self.player.rollback()
        elif status == 2:
            if not self.player.has_key:
                reward = 10
                self.player.has_key = True
                self.key = None
        elif status == 3:
            if self.player.has_key:
                reward = 20
                self.car = None
            else:
                self.player.rollback()
        self.score += reward
        return reward
    
    def draw(self):
        self.screen.fill(BG_COLOR)
        for wall in self.walls:
            pg.draw.rect(self.screen, WHITE, list(wall) + [GRID_LEN] * 2)
        if self.key:
            self.key.draw(self.screen)
        if self.car:
            self.car.draw(self.screen)
        self.player.draw(self.screen)

    def keydown(self, direc):
        self.player.set_direction(direc)

    def init_render(self):
        self.screen = pg.display.set_mode(WINDOW_SIZE)
        pg.display.set_caption('Navigate v1.0.0')
        self.display = True

    def render(self):
        pg.display.flip()


class NaviEnv:
    def __init__(self):
        pg.init()
        self.game = Game()
    
    def reset(self):
        self.game.reset()
        self.game.draw()
        observe = pg.surfarray.array3d(self.game.screen)
        return observe
        
    def step(self, action):
        if self.game.display:
            pg.event.pump()
        self.game.keydown(action)
        reward = self.game.update()
        done = (not self.game.car)
        self.game.draw()
        observe = pg.surfarray.array3d(self.game.screen)
        info = {'pos': self.game.player.rect.topleft}
        return observe, reward, done, info

    def init_render(self):
        self.game.init_render()

    def render(self):
        if not self.game.display:
            self.init_render()
        pg.display.flip()


if __name__ == '__main__':
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (64, 64)
    clock = pg.time.Clock()
    render = True
    game = Game()
    if render:
        game.init_render()
    step = 0
    while game.car is not None:
        time.sleep(0.5)
        step += 1
        clock.tick(30)
        for evt in pg.event.get():
            if evt.type == pg.QUIT:
                quit()
            elif evt.type == pg.KEYDOWN:
                if evt.key == pg.K_UP:
                    game.keydown(UP)
                elif evt.key == pg.K_DOWN:
                    game.keydown(DOWN)
                elif evt.key == pg.K_LEFT:
                    game.keydown(LEFT)
                elif evt.key == pg.K_RIGHT:
                    game.keydown(RIGHT)
                elif evt.key == pg.K_ESCAPE:
                    quit()

        game.update()
        game.draw()
        if render:
            game.render()            

    print('Score:', game.score, 'Step:', step)

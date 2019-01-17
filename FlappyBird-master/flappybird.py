#!/usr/bin/env python

import pygame
from pygame.locals import *  # noqa
import sys
import random
import numpy as np

Y_CHANGE_FROM_ACTION = {0: 0,
                        1: -10}


class FlappyBird:
    def __init__(self, max_gens):

        self.max_gens = max_gens
        self.gen_counter = 0
        self.step_counter = 0
        self.random_factor = 0.0
        self.gamma = 0.5
        self.learning_rate = 0.1

        self.screen = pygame.display.set_mode((400, 708))
        self.bird = pygame.Rect(65, 50, 50, 50)
        self.background = pygame.image.load("assets/background.png").convert()
        self.birdSprites = [pygame.image.load("assets/1.png").convert_alpha(),
                            pygame.image.load("assets/2.png").convert_alpha(),
                            pygame.image.load("assets/dead.png")]
        self.wallUp = pygame.image.load("assets/bottom.png").convert_alpha()
        self.wallDown = pygame.image.load("assets/top.png").convert_alpha()
        self.gap = 130
        self.wallx = 400
        self.birdY = 350
        self.birdX = 70
        self.jump = 0
        self.jumpSpeed = 10
        self.gravity = 5
        self.dead = False
        self.sprite = 0
        self.counter = 0
        self.offset = random.randint(-110, 110)

        self.centerdot = pygame.Surface((5, 5))
        self.centerdot.fill((0, 0, 0))

        # Dim: action, dy, dx
        self.R = np.zeros((2, 120, 48))

        # Add reward when bird is in middle of gap and jumps while passing walls
        self.R[1, self.dy_to_index(0), self.dx_to_index(0)] = 100

        # Add punishment when hitting wall inside gap
        for x in range(0, 80):
            xindex = self.dx_to_index(x)

            self.R[:, self.dy_to_index(-64), xindex] = -100

            self.R[:, self.dy_to_index(65), xindex] = -100

        # self.Q = np.zeros((2, 120, 48))
        self.Q = np.random.rand(2, 120, 48)
        self.Q[1, :, :] *= 0.1

    def train_run(self):

        pygame.font.init()
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 50)
        while self.gen_counter < self.max_gens:
            self.step_counter += 1

            clock.tick(200)

            x_index = self.dx_to_index(self.bird_wall_dist())
            y_index = self.dy_to_index(self.delta_y())

            # print('delta y :', self.delta_y(), 'yindex: ', y_index)
            if self.delta_y() < self.gap / 2 and not self.dead:

                # Random chance to pick random move, otherwise pick max Q
                action = None
                if random.uniform(0, 1) < self.random_factor:
                    # Take action
                    action = random.randint(0, 1)
                    if self.R[action, y_index, x_index] == -1:
                        continue

                    else:
                        action_list = [action]
                        print('Random action: ', action_list)

                else:
                    max_Q = np.max(self.Q[:, y_index, x_index])
                    action_list = np.argwhere(self.Q[:, y_index, x_index] == max_Q).flatten().tolist()

                    # action_list = random.sample(action_list, len(action_list))
                    print('non random actions are ', action_list)

                print('going to state change')
                state_change = None
                for potential_action in action_list:
                    action = potential_action

                    if self.R[action, y_index, x_index] == -1:
                        continue

                    else:
                        state_change = Y_CHANGE_FROM_ACTION[action]
                        break

                if self.R[action, y_index, x_index] == -1:
                    continue

                self.birdY += state_change
                new_y_index = self.dy_to_index(self.delta_y())

                max_Q = np.max(self.Q[:, new_y_index, x_index])

                self.Q[action, new_y_index, x_index] = (1 - self.learning_rate) * self.Q[action, new_y_index, x_index] + \
                                                       self.learning_rate * (self.R[
                                                                                 action, new_y_index, x_index] + self.gamma * max_Q)

                print('updating {}, {}, {} to '.format(action, new_y_index, x_index), ' to ',
                      self.Q[action, new_y_index, x_index])

                if action:
                    self.jump = 17
                    self.gravity = 5
                    self.jumpSpeed = 10

            self.screen.fill((255, 255, 255))
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.wallUp,
                             (self.wallx, 360 + self.gap - self.offset))
            self.screen.blit(self.wallDown,
                             (self.wallx, 0 - self.gap - self.offset))
            self.screen.blit(font.render(str(self.counter),
                                         -1,
                                         (255, 255, 255)),
                             (200, 50))
            if self.dead:
                self.sprite = 2
            elif self.jump:
                self.sprite = 1
            self.screen.blit(self.birdSprites[self.sprite], (self.birdX, self.birdY))
            if not self.dead:
                self.sprite = 0
            self.updateWalls()
            self.birdUpdate()

            # Draw center of gap
            self.screen.blit(self.centerdot, (self.wallx,
                                              0 - self.gap / 2 + 500 - self.offset))
            # Draw birdY
            self.screen.blit(self.centerdot, (self.wallx,
                                              self.birdY))

            pygame.display.update()

    def bird_wall_dist(self):
        return self.wallx - self.birdX - 44

    def delta_y(self):
        gap_center = 0 - self.gap / 2 + 500 - self.offset
        return gap_center - self.birdY

    def dx_to_index(self, dx):
        xmin = -193.9
        return int(dx - xmin) // 10

    def dy_to_index(self, dy):
        ymin = -599.9
        return int(dy - ymin) // 10

    def updateWalls(self):
        self.wallx -= 2
        if self.wallx < -80:
            self.wallx = 400
            self.counter += 1
            # self.offset = random.randint(-110, 110)
            self.offset = 0

    def birdUpdate(self):
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            self.birdY += self.gravity
            self.gravity += 0.2
        self.bird[1] = self.birdY
        upRect = pygame.Rect(self.wallx,
                             360 + self.gap - self.offset + 10,
                             self.wallUp.get_width() - 10,
                             self.wallUp.get_height())
        downRect = pygame.Rect(self.wallx,
                               0 - self.gap - self.offset - 10,
                               self.wallDown.get_width() - 10,
                               self.wallDown.get_height())
        if upRect.colliderect(self.bird):
            print('died')
            self.dead = True
        if downRect.colliderect(self.bird):
            print('died')
            self.dead = True
        if not 0 < self.bird[1] < 720:
            self.bird[1] = 50
            self.birdY = 50
            self.dead = False
            self.counter = 0
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.gravity = 5

            print(self.Q)

    def state_monitor(self):
        gap_center = 0 - self.gap / 2 + 500 - self.offset
        return [gap_center, self.delta_y(), self.birdY + 15, self.bird_wall_dist()]

    def run(self):
        clock = pygame.time.Clock()
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 50)
        while True:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
                    self.jump = 17
                    self.gravity = 5
                    self.jumpSpeed = 10

            self.screen.fill((255, 255, 255))
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.wallUp,
                             (self.wallx, 360 + self.gap - self.offset))
            self.screen.blit(self.wallDown,
                             (self.wallx, 0 - self.gap - self.offset))
            self.screen.blit(font.render(str(self.counter),
                                         -1,
                                         (255, 255, 255)),
                             (200, 50))
            if self.dead:
                self.sprite = 2
            elif self.jump:
                self.sprite = 1
            self.screen.blit(self.birdSprites[self.sprite], (self.birdX, self.birdY))
            if not self.dead:
                self.sprite = 0
            self.updateWalls()
            self.birdUpdate()

            if not self.dead:
                print(self.state_monitor())

            # Draw center of gap
            self.screen.blit(self.centerdot, (self.wallx,
                                              0 - self.gap / 2 + 500 - self.offset))
            # Draw birdY
            self.screen.blit(self.centerdot, (self.wallx,
                                              self.birdY))

            pygame.display.update()


if __name__ == "__main__":
    fb = FlappyBird(max_gens=10)

    # fb.run()
    fb.train_run()

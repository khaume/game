import pygame
import sys
import random


black = 0, 0, 0
red = 200, 0, 0
blue = 0, 20, 200
green = 0, 200, 0

class FlappyBird:
    def __init__(self):

        pygame.init()
        pygame.font.init()

        self.size = self.width, self.height = 400, 708
        self.screen = pygame.display.set_mode(self.size)
        self.background = pygame.image.load("assets/background.png").convert()

        self.birdsprites = [pygame.image.load("assets/1.png"),
                            pygame.image.load("assets/2.png"),
                            pygame.image.load("assets/dead.png")]

        self.wallsprites = [pygame.image.load("assets/bottom.png"),
                            pygame.image.load("assets/top.png")]

        self.bird = pygame.Surface((30, 30))
        self.bird.fill(green)

        self.gap = 120
        self.wallheight = 500

        self.wallx = self.width-100

        self.birdy = self.height/2

        self.center = self.height / 2

        self.walltopup = self.center - self.gap / 2 - self.wallheight
        self.walltopdown = self.center + self.gap/2

        self.gravity = 5
        self.jump = 17
        self.jumpspeed = 10

    def update_walls(self):
        self.wallx -= 2
        if self.wallx < - 100:
            self.wallx = self.width

            self.center = random.randint(250, 550)

            self.walltopup = self.center - self.gap / 2 - self.wallheight
            self.walltopdown = self.center + self.gap / 2



    def update_bird(self):
        if self.jump:
            self.jumpspeed -= 1
            self.birdy -= self.jumpspeed
            self.jump -= 1
        else:
            self.birdy += self.gravity
            self.gravity += 0.2

        if self.birdy > self.height - 20:
            self.birdy = self.height/2
            self.gravity = 0.5


    def run(self):
        clock = pygame.time.Clock()
        while True:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    self.jump = 17
                    self.gravity = 5
                    self.jumpspeed = 10


            self.screen.fill(black)
            self.screen.blit(self.background, (0, 0))

            self.screen.blit(self.wallsprites[1], (self.wallx, self.walltopup))
            self.screen.blit(self.wallsprites[0], (self.wallx, self.walltopdown))

            if self.jump:
                self.screen.blit(self.birdsprites[1], (self.width/2, self.birdy))
            else:
                self.screen.blit(self.birdsprites[0], (self.width/2, self.birdy))

            self.update_bird()
            self.update_walls()

            pygame.display.update()

if __name__ == '__main__':
    FlappyBird().run()

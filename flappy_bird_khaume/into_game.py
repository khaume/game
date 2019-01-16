import pygame
import sys
import random


black = 0, 0, 0
red = 200, 0, 0
blue = 0, 20, 200
green = 0, 200, 0

class Snake:
    def __init__(self):

        pygame.init()
        pygame.font.init()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 30)

        self.size = self.width, self.height = 820, 640
        self.speed = [0, 0]

        self.screen = pygame.display.set_mode(self.size)
# ball = pygame.image.load('intro_ball.gif')

        self.score = 0
        self.textsurface = self.myfont.render(str(self.score), False, (200, 200, 200))

        self.ball = pygame.Surface((30, 30))
        self.ball.fill(green)


        self.ballrect = self.ball.get_rect()

        self.obs_posx = 40
        self.obs_posy = 40
        self.obs_size = 30

        self.obstacle = pygame.Surface((self.obs_size, self.obs_size))
        self.obstacle.fill(red)
        self.screen.blit(self.obstacle, (self.obs_posx, self.obs_posy))



    def detect_collision(self):
        if (self.ballrect.left < self.obs_posx + self.obs_size and self.ballrect.right > self.obs_posx and
            self.ballrect.top < self.obs_posy + self.obs_size and self.ballrect.bottom > self.obs_posy):
            self.obs_posx = random.randint(self.obs_size, self.width - self.obs_size)
            self.obs_posy = random.randint(self.obs_size, self.height - self.obs_size)

            self.score += 1
            self.textsurface = self.myfont.render(str(self.score), False, (200, 200, 200))

    # if ballrect.top == obs_posy or ballrect.bottom == obs_posy:
    #     screen.blit(obstacle, (obs_posx+10, obs_posy))

    def update_ball(self):
        self.ballrect = self.ballrect.move(self.speed)
        # self.speed = [0, 0]

        if self.ballrect.left < 0 or self.ballrect.right > self.width:
            self.speed[0] = -self.speed[0]

        if self.ballrect.top < 0 or self.ballrect.bottom > self.height:
            self.speed[1] = -self.speed[1]


    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # playing = False
                        sys.exit()
                    if event.key == pygame.K_w:
                        self.speed[1] = -1
                        self.speed[0] = 0

                    if event.key == pygame.K_s:
                        self.speed[1] = 1
                        self.speed[0] = 0

                    if event.key == pygame.K_a:
                        self.speed[0] = -1
                        self.speed[1] = 0

                    if event.key == pygame.K_d:
                        self.speed[0] = 1
                        self.speed[1] = 0


            self.screen.fill(black)
            self.update_ball()
            self.detect_collision()

            self.screen.blit(self.ball, self.ballrect)
            self.screen.blit(self.obstacle, (self.obs_posx, self.obs_posy))
            self.screen.blit(self.textsurface, (30, 30))
            pygame.display.update()

if __name__ == '__main__':
    Snake().run()

import numpy as np
import pygame
import random
import sys
import matplotlib.pyplot as plt

# Global colors
BLACK = 0, 0, 0
WHITE = 255, 255, 255
RED = 255, 0, 0
BLUE = 0, 0, 255
GREEN = 0, 255, 0

STATE_CHANGE_FROM_ACTION = {0: -5,
                            1: 5,
                            2: -1,
                            3: 1}


class GridMove:
    def __init__(self, obs_1_state, obs_2_state, goal_state, max_gens):
        pygame.init()

        self.obs_1_state = obs_1_state
        self.obs_2_state = obs_2_state
        self.goal_state = goal_state
        self.max_gens = max_gens
        self.gen_counter = 0
        self.step_counter = 0

        self.performance_array = np.zeros((self.max_gens, 1))

        # Set screen
        self.width, self.height = 500, 500
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.background = pygame.image.load('background.png')

        # Set player
        self.player = pygame.image.load('0.png')
        self.player_position = [30, 30]

        # Set obstacles
        self.obstacle = [pygame.Surface((100, 100)),
                         pygame.Surface((100, 100))]
        self.obstacle[0].fill(RED)
        self.obstacle[1].fill(BLUE)
        self.obstacle_position = [self.location_from_state(obs_1_state, field=True),
                                  self.location_from_state(obs_2_state, field=True)]

        # Set goal
        self.goal = pygame.Surface((100, 100))
        self.goal.fill(GREEN)
        self.goal_state = goal_state
        self.goal_position = self.location_from_state(goal_state, field=True)

        self.score = 0
        self.gamma = 0.5
        self.learning_rate = 0.1
        self.random_factor = 0.1

        # Set reward and Q matrices
        self.Q = np.zeros((25, 4))
        self.R = np.zeros((25, 4))

        # Adding rewards for goal and obstacles.
        for key in STATE_CHANGE_FROM_ACTION.keys():
            mat_len = self.R.shape[0]

            # For goal.
            if (0 <= goal_state - STATE_CHANGE_FROM_ACTION[key]) and \
                    (goal_state - STATE_CHANGE_FROM_ACTION[key] < mat_len):
                self.R[goal_state - STATE_CHANGE_FROM_ACTION[key], key] = 100

            # For obstacles.
            for obs_state in [obs_1_state, obs_2_state]:

                if (0 <= obs_state - STATE_CHANGE_FROM_ACTION[key]) and \
                        (obs_state - STATE_CHANGE_FROM_ACTION[key] < mat_len):
                    self.R[obs_state - STATE_CHANGE_FROM_ACTION[key], key] = -100

        # For sides
        # Top
        for i in range(5):
            self.R[i, 0] = -1
        # Left
        for i in range(0, 25, 5):
            self.R[i, 2] = -1
        # Right
        for i in range(4, 25, 5):
            self.R[i, 3] = -1
        # Bottom
        for i in range(20, 25):
            self.R[i, 1] = -1

        # print(self.R)

    def state_from_location(self, position):
        player_index = [(position[0] - 30) // 100,
                        (position[1] - 30) // 100]

        player_state = (player_index[1] + 1) * 5 - (5 - player_index[0])

        return player_state

    def location_from_state(self, state, field=False):
        player_index = [state % 5,
                        state // 5]

        position = [player_index[0] * 100 + 30,
                    player_index[1] * 100 + 30]

        if field:
            position[0] -= 30
            position[1] -= 30

        return position

    def move(self, direction):
        # Keypress w, up
        if direction == 0 and self.player_position[1] > 100:
            self.player_position[1] -= 100

        # Keypress s, down
        elif direction == 1 and self.player_position[1] < 400:
            self.player_position[1] += 100

        # Keypress a, left
        elif direction == 2 and self.player_position[0] > 100:
            self.player_position[0] -= 100

        # Keypress d, right
        elif direction == 3 and self.player_position[0] < 400:
            self.player_position[0] += 100

    def check_win_loss(self):
        if (self.player_position[0] - 30 == self.goal_position[0] and
                self.player_position[1] - 30 == self.goal_position[1]):
            # print(self.Q)

            self.player_position[0] = 30
            self.player_position[1] = 30
            self.gen_counter += 1

            self.performance_array[self.gen_counter - 1] = self.step_counter

            print('win with generation ', self.gen_counter, ' after steps: ', self.step_counter)

            self.step_counter = 0

        if (self.player_position[0] - 30 == self.obstacle_position[0][0] and
                self.player_position[1] - 30 == self.obstacle_position[0][1]):
            print('lose')
            # print(self.Q)

            self.player_position[0] = 30
            self.player_position[1] = 30
            # self.gen_counter -= 1

        if (self.player_position[0] - 30 == self.obstacle_position[1][0] and
                self.player_position[1] - 30 == self.obstacle_position[1][1]):
            print('lose')
            # print(self.Q)

            self.player_position[0] = 30
            self.player_position[1] = 30
            # self.gen_counter -= 1

    def train(self, iter):
        counter = 0

        clock = pygame.time.Clock()
        while counter < iter:
            clock.tick(10)

            player_state = self.state_from_location(self.player_position)
            print('player state is ', player_state)

            # Take action
            action = random.randint(0, 3)
            print('Action: ', action)

            # Encode action to state change
            state_change = STATE_CHANGE_FROM_ACTION[action]

            print('matrix: ', self.R[player_state][action])

            # Find next state
            if self.R[player_state][action] == -1:
                continue

            else:
                # print('R of action: ', self.R[player_state][action])

                next_state = player_state + state_change
                # print('Next state: ', next_state)

                max_Q = np.max(self.Q[next_state])
                # print('Max Q: ', max_Q)

                self.Q[player_state][action] = self.R[player_state][action] + self.gamma * max_Q
                # print('Updating Q[{}][{}] to: '.format(player_state, action), self.R[player_state][action] + self.gamma * max_Q)

                self.move(action)
                player_state = self.state_from_location(self.player_position)

            # if player_state in [self.goal_state, self.obs_1_state, self.obs_2_state]:
            #     player_state = 0

            self.player_position = self.location_from_state(player_state)

            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.obstacle[0], (self.obstacle_position[0]))
            self.screen.blit(self.obstacle[1], (self.obstacle_position[1]))
            self.screen.blit(self.goal, self.goal_position)
            self.screen.blit(self.player, (self.player_position[0], self.player_position[1],))

            pygame.display.update()

            counter += 1

    def train_run(self):

        clock = pygame.time.Clock()
        while self.gen_counter < self.max_gens:
            self.step_counter += 1

            clock.tick(80)

            player_state = self.state_from_location(self.player_position)

            # Random chance to pick random move, otherwise pick max Q
            action = None
            if random.uniform(0, 1) < self.random_factor:
                # Take action
                action = random.randint(0, 3)
                if self.R[player_state][action] == -1:
                    continue

                else:
                    action_list = [action]
                    print('Random action: ', action_list)

            else:
                max_Q = np.max(self.Q[player_state])
                action_list = np.argwhere(self.Q[player_state] == max_Q).flatten().tolist()
                # print('max Q: ', max_Q)

                action_list = random.sample(action_list, len(action_list))
                print('non random actions are ', action_list)

            state_change = None
            for potential_action in action_list:
                action = potential_action

                # print('considering action: ', action)
                # print('reward here is ', self.R[player_state][action])

                if self.R[player_state][action] == -1:
                    continue

                else:
                    state_change = STATE_CHANGE_FROM_ACTION[action]
                    break

            if self.R[player_state][action] == -1:
                continue

            next_state = player_state + state_change
            # print('Next state: ', next_state)

            max_Q = np.max(self.Q[next_state])
            # print('Max Q: ', max_Q)

            self.Q[player_state][action] = (1 - self.learning_rate) * self.Q[player_state][action] + \
                                           self.learning_rate * (self.R[player_state][action] + self.gamma * max_Q)

            # self.Q[player_state][action] = self.R[player_state][action] + self.gamma * max_Q
            # print('Updating Q[{}][{}] to: '.format(player_state, action), self.R[player_state][action] + self.gamma * max_Q)

            self.move(action)
            player_state = self.state_from_location(self.player_position)
            self.check_win_loss()

            # if player_state in [self.goal_state, self.obs_1_state, self.obs_2_state]:
            #     player_state = 0

            # self.player_position = self.location_from_state(player_state)

            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.obstacle[0], (self.obstacle_position[0]))
            self.screen.blit(self.obstacle[1], (self.obstacle_position[1]))
            self.screen.blit(self.goal, self.goal_position)
            self.screen.blit(self.player, (self.player_position[0], self.player_position[1],))

            pygame.display.update()

            # iter_counter += 1

    def play(self):
        clock = pygame.time.Clock()
        while True:
            clock.tick(10)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        sys.exit()

                    if event.key == pygame.K_w:
                        self.move(0)

                    if event.key == pygame.K_s:
                        self.move(1)

                    if event.key == pygame.K_a:
                        self.move(2)

                    if event.key == pygame.K_d:
                        self.move(3)

                    self.check_win_loss()

                    self.screen.blit(self.background, (0, 0))
                    self.screen.blit(self.obstacle[0], (self.obstacle_position[0]))
                    self.screen.blit(self.obstacle[1], (self.obstacle_position[1]))
                    self.screen.blit(self.goal, self.goal_position)
                    self.screen.blit(self.player, (self.player_position[0], self.player_position[1],))

                    pygame.display.update()

                    player_state = self.state_from_location(self.player_position)

                    print('State: ', player_state)

    def bot_run(self):
        print('running the bot')
        self.player_position = [30, 30]

        clock = pygame.time.Clock()
        while True:
            clock.tick(10)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    # print('key down registered')
                    if event.key == pygame.K_ESCAPE:
                        sys.exit()

                    if event.key == pygame.K_SPACE:
                        # print('printed SPACE')

                        player_state = self.state_from_location(self.player_position)

                        max_Q = np.max(self.Q[player_state])
                        action_list = np.argwhere(self.Q[player_state] == max_Q).flatten().tolist()
                        # print('max Q: ', max_Q)

                        action_list = random.sample(action_list, len(action_list))
                        # print('possible actions are ', action_list)

                        state_change = None

                        for action in action_list:
                            # print('considering action: ', action)
                            # print('reward here is ', self.R[player_state][action])

                            if self.R[player_state][action] == -1:
                                continue

                            else:
                                state_change = STATE_CHANGE_FROM_ACTION[action]
                                break

                        # print(self.Q)
                        # print('state was ', player_state, 'action was ', action, ' state change is ', state_change)
                        player_state += state_change

                        self.check_win_loss()

                        self.player_position = self.location_from_state(player_state)

                        self.screen.blit(self.background, (0, 0))
                        self.screen.blit(self.obstacle[0], (self.obstacle_position[0]))
                        self.screen.blit(self.obstacle[1], (self.obstacle_position[1]))
                        self.screen.blit(self.goal, self.goal_position)
                        self.screen.blit(self.player, (self.player_position[0], self.player_position[1],))

                        pygame.display.update()


if __name__ == '__main__':
    gm = GridMove(obs_1_state=1,
                  obs_2_state=8,
                  goal_state=24,
                  max_gens=50)

    gm.train_run()
    plt.plot(range(len(gm.performance_array)), gm.performance_array)
    plt.show()

    # gm.bot_run()
    # gm.play()

    # print(np.around(gm.Q, 3))
    # print(gm.performance_array)

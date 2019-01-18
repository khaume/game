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

STATE_NUM = 25


class GridMove:
    def __init__(self, obs_1_state, obs_2_state, goal_state, max_gens):

        # Set obstacles and goal.
        self.obs_1_state = obs_1_state
        self.obs_2_state = obs_2_state
        self.goal_state = goal_state

        # Set training variables. max_gens is the number of generations to train on.
        # Gamma and learning_rate are from the Bellman equation, see https://en.wikipedia.org/wiki/Q-learning#Algorithm
        # Random factor is the fraction of moves which will not follow the greedy, optimal route, but take a random
        # action.
        self.max_gens = max_gens
        self.gamma = 0.5
        self.learning_rate = 0.1
        self.random_factor = 0.1

        # Counters to keep track of generation and number of steps taken.
        self.gen_counter = 0
        self.step_counter = 0

        # Array for recording performance for each generation.
        self.performance_array = np.zeros((self.max_gens, 1))
        self.history_array = np.zeros((STATE_NUM, 1))

        # Set screen.
        self.width, self.height = 500, 500
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.background = pygame.image.load('background.png')

        # Set player.
        self.player = pygame.image.load('0.png')
        self.player_position = [30, 30]

        # Set obstacles.
        self.obstacle = [pygame.Surface((100, 100)),
                         pygame.Surface((100, 100))]
        self.obstacle[0].fill(RED)
        self.obstacle[1].fill(BLUE)
        self.obstacle_positions = [self.location_from_state(obs_1_state, field=True),
                                   self.location_from_state(obs_2_state, field=True)]

        # Set goal.
        self.goal = pygame.Surface((100, 100))
        self.goal.fill(GREEN)
        self.goal_position = self.location_from_state(goal_state, field=True)

        # Initialize reward and Q matrices.
        self.Q = np.zeros((25, 4))
        self.R = np.zeros((25, 4))

        # Adding rewards for goal and obstacles.
        for key in STATE_CHANGE_FROM_ACTION.keys():

            # For goal.
            # Make all the states around the goal give 100 reward for actions that lead to the goal state, but only
            # if they are inside the board (between [0 ; STATE_NUM[
            if (0 <= goal_state - STATE_CHANGE_FROM_ACTION[key]) and \
                    (goal_state - STATE_CHANGE_FROM_ACTION[key] < STATE_NUM):
                self.R[goal_state - STATE_CHANGE_FROM_ACTION[key], key] = 100

            # For obstacles.
            # Like for goal but -100 for states around the obstacles.
            for obs_state in [obs_1_state, obs_2_state]:

                if (0 <= obs_state - STATE_CHANGE_FROM_ACTION[key]) and \
                        (obs_state - STATE_CHANGE_FROM_ACTION[key] < STATE_NUM):
                    self.R[obs_state - STATE_CHANGE_FROM_ACTION[key], key] = -100

        # Set reward of -1 for actions that lead outside the board.
        # Top.
        for i in range(5):
            self.R[i, 0] = -1
        # Left.
        for i in range(0, 25, 5):
            self.R[i, 2] = -1
        # Right.
        for i in range(4, 25, 5):
            self.R[i, 3] = -1
        # Bottom.
        for i in range(20, 25):
            self.R[i, 1] = -1

    def state_from_location(self, position):
        # Get the state from a position.
        # First find the matrix indices on the board. Example: 230, 330 --> 2, 3
        player_index = [(position[0] - 30) // 100,
                        (position[1] - 30) // 100]

        # Then find the state number from the indices. Example 2, 3 --> 17
        player_state = player_index[1] * 5 + player_index[0]

        return player_state

    def location_from_state(self, state, field=False):
        # Find the location from a state.
        # Example: 17 --> 2, 3
        player_index = [state % 5,
                        state // 5]

        # Example: 2, 3 --> 230, 330
        position = [player_index[0] * 100 + 30,
                    player_index[1] * 100 + 30]

        # If field=True, then we are finding the position for an obstacle or the goal, and we want the upper
        # left corner of the state, so we subtract 30, 30
        if field:
            position[0] -= 30
            position[1] -= 30

        return position

    def move(self, direction):
        # Adjust the player_position after making a move, if that will not move the player outside the board.

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

    def record_state(self):
        self.history_array[self.state_from_location(self.player_position)] += 1

    def check_win_loss(self):
        # Check if the player is on a goal or obstacle.

        # If on goal.
        if self.state_from_location(self.player_position) == self.goal_state:
            # Reset player position and increment the counter and update performance array
            self.player_position = self.location_from_state(0)
            self.gen_counter += 1
            self.performance_array[self.gen_counter - 1] = self.step_counter

            print('win with generation ', self.gen_counter, ' after steps: ', self.step_counter)

            # Reset step_counter.
            self.step_counter = 0

        # If on obstacles.
        for obstacle_state in [self.obs_1_state, self.obs_2_state]:
            if self.player_position == self.location_from_state(obstacle_state):
                # Reset player position.
                self.player_position = self.location_from_state(0)

    def train(self, iteration):

        clock = pygame.time.Clock()
        while self.gen_counter < self.max_gens:
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
                self.check_win_loss()
                player_state = self.state_from_location(self.player_position)

            self.player_position = self.location_from_state(player_state)

            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.obstacle[0], (self.obstacle_positions[0]))
            self.screen.blit(self.obstacle[1], (self.obstacle_positions[1]))
            self.screen.blit(self.goal, self.goal_position)
            self.screen.blit(self.player, (self.player_position[0], self.player_position[1],))

            pygame.display.update()


    def train_run(self):
        # In this method, the Q learning algorithm will update the Q matrix while the player moves around in a
        # greedy manner. To begin with it is random, but quickly the Q matrix will provide a preference in the
        # direction. self.random_factor provides a fraction of the moves that is random, to help escaping local
        # minima.

        clock = pygame.time.Clock()
        while self.gen_counter < self.max_gens:
            # Set the FPS of the game.
            clock.tick(80)

            self.step_counter += 1

            player_state = self.state_from_location(self.player_position)

            # Chance to pick random move, otherwise pick max Q.
            action = None
            action_list = []
            print('about to find move')
            if random.uniform(0, 1) < self.random_factor:
                # Take random action.
                action = random.randint(0, 3)
                if self.R[player_state][action] == -1:
                    # If move would lead outside board, go opposite direction.
                    action = 3 - action

                else:
                    action_list = [action]
                    print('Random action: ', action_list)

            else:
                # Find the max Q value, meaning for the current state, what is the maximum value possible  taking any
                # action.
                max_Q = np.max(self.Q[player_state])

                # Find the moves which have this value. We make a list of these moves, because there may be more
                # than one move which has same max Q value.
                # This is especially the case in the beginning when max Q is zero almost everywhere.
                action_list = np.argwhere(self.Q[player_state] == max_Q).flatten().tolist()

                # Shuffle the list to avoid always picking the lowest action.
                action_list = random.sample(action_list, len(action_list))
                print('non random actions are ', action_list)

            state_change = None
            # Evaluate the potential moves from the action_list.
            for potential_action in action_list:
                print('trying action: ', potential_action, 'from state ', player_state)
                action = potential_action

                # print('considering action: ', action)
                # print('reward here is ', self.R[player_state][action])

                if self.R[player_state][action] == -1:
                    print('bad move')
                    continue

                else:
                    print('setting state change')
                    state_change = STATE_CHANGE_FROM_ACTION[action]
                    break

            if self.R[player_state][action] == -1:
                continue

            if state_change is None:
                print('setting state change manually')
                state_change = STATE_CHANGE_FROM_ACTION[action]

            next_state = player_state + state_change
            # print('Next state: ', next_state)

            max_Q = np.max(self.Q[next_state])
            # print('Max Q: ', max_Q)

            self.Q[player_state][action] = (1 - self.learning_rate) * self.Q[player_state][action] + \
                                           self.learning_rate * (self.R[player_state][action] + self.gamma * max_Q)

            print('Updating Q[{}][{}] to: '.format(player_state, action),
                  self.R[player_state][action] + self.gamma * max_Q)

            self.move(action)
            self.check_win_loss()
            self.record_state()

            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.obstacle[0], (self.obstacle_positions[0]))
            self.screen.blit(self.obstacle[1], (self.obstacle_positions[1]))
            self.screen.blit(self.goal, self.goal_position)
            self.screen.blit(self.player, (self.player_position[0], self.player_position[1],))

            pygame.display.update()

    def play(self):
        clock = pygame.time.Clock()
        while True:
            # Set the framerate to 10 FPS.
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
                    self.screen.blit(self.obstacle[0], (self.obstacle_positions[0]))
                    self.screen.blit(self.obstacle[1], (self.obstacle_positions[1]))
                    self.screen.blit(self.goal, self.goal_position)
                    self.screen.blit(self.player, (self.player_position[0], self.player_position[1],))

                    self.state_from_location(self.player_position)

                    pygame.display.update()


if __name__ == '__main__':
    gm = GridMove(obs_1_state=1,
                  obs_2_state=8,
                  goal_state=24,
                  max_gens=50)

    gm.train_run()

    plt.plot(range(len(gm.performance_array)), gm.performance_array)
    plt.show()

    plt.imshow(gm.history_array.reshape((5, 5)), cmap='hot', interpolation='nearest')
    plt.show()
    # gm.play()

    # print(np.around(gm.Q, 3))
    # print(gm.performance_array)

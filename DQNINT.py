import numpy as np
from collections import deque
import random
import math
import skimage

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import load_model

from GameHandler import Game

# Parameters
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
ENV_LOCATION = "build/UnityFrogge"
# rewards for terminal find
DEATH_REWARD = -50
GAME_OVER_REWARD = -50
GOAL_REWARD = 50

# model parameters
LEARNING_RATE = 0.00025
STACK_SIZE = 4  # stack size for single state
BATCH = 64  # size of mini batch
GAMMA = 0.99

# agent
FINAL_EPSILON = 0.1  # final value of epsilon
INITIAL_EPSILON = 1  # starting value of epsilon
OBSERVER = 100  #50000  # filling D (experience replay data)
REPLAY_SIZE = 100000  # size of D

#
TOTAL_EPI = 9000000
C = 10000  # update q`

# soft max temp var
TEMPERATURE = 1


# -- Brain --#
class Brain:

    def __init__(self):
        self.model = self._create_model()   # q model
        self._model = self._create_model()  # q` model ( used to calculate predication for error)
        self.training_loss = 0

    # no init of network param : we can use normal dist.-:init=lambda shape, name: normal(shape, scale=0.01, name=name),
    def _create_model(self):

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, STACK_SIZE)))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('linear'))
        # model.add(Activation('relu'))
        model.add(Dense(5))

        opt = RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=0.01)
        # opt = Adam(lr=LEARNING_RATE)
        model.compile(loss='mean_squared_error', optimizer=opt)  # mse for dqn

        return model

    # predict action from state images
    def predict_action(self, state):

        # todo : cal soft max and call lan based model to find the action
        q = self.model.predict(self.pre_process_images(state))  # input a stack of 4 images, get the prediction
        max_q = np.argmax(q)
        action_val = max_q

        return action_val

    # calculate softmax values for actions
    # used fixed 5 actions
    def calculate_softmax(self, q):
        softmax_val = []
        # cal total softmax
        total = 0
        for i in range(0, 5):
            total += math.exp(q[i]/TEMPERATURE)
        # cal each soft max
        for i in range(0, 5):
            softmax_val.append((math.exp(q[i]/TEMPERATURE)/total))

        return softmax_val

    # train model using the re play queue
    def train(self, mini_batch):

        self.training_loss = 0

        inputs = np.zeros((BATCH, IMAGE_HEIGHT, IMAGE_WIDTH, STACK_SIZE))  # 2500, 100, 100, 4
        targets = np.zeros((inputs.shape[0], 5))  # 2500, 4
        targets_ = np.zeros((inputs.shape[0], 5))  # 2500, 4

        # Now we do the experience replay
        for j in range(0, len(mini_batch)):
            state_t = mini_batch[j][0]  # state
            action_t = mini_batch[j][1]  # action
            reward_t = mini_batch[j][2]  # reward
            state_t1 = mini_batch[j][3]  # new state
            terminal = mini_batch[j][4]  # is terminal reached or not

            inputs[j:j + 1] = self.pre_process_images(state_t)  # saved down s_t as input

            # predict q values for current state
            targets[j] = self.model.predict(self.pre_process_images(state_t))
            # predict q values for next state
            targets_[j] = self.model.predict(self.pre_process_images(state_t1))

            # todo : modification to call lan based model and predict actions for the training
            softmax_values = self.calculate_softmax(targets[j])

            q_sa = self._model.predict(self.pre_process_images(state_t1))  # predict to get arg max Q to cal TD

            if terminal:
                targets[j, action_t] = reward_t  # if terminal only set target as reward for the action
            else:
                targets[j, action_t] = reward_t + GAMMA * q_sa[0][np.argmax(targets_[j])]

        self.training_loss += self.model.train_on_batch(inputs, targets)

    # method to convert images to B/W
    def pre_process_images(self, state):
        size = (IMAGE_WIDTH, IMAGE_HEIGHT, STACK_SIZE)  # create list to keep frames
        stack = np.zeros(size)

        for i in range(0, STACK_SIZE):
            st = skimage.color.rgb2gray(state[0][:, :, :, i])
            st_gray = skimage.transform.resize(st, (IMAGE_WIDTH, IMAGE_HEIGHT))
            stack[:, :, i] = st_gray

        stack = stack.reshape(1, stack.shape[0], stack.shape[1], stack.shape[2])

        return stack

    # update q`
    def update_target_model(self):
        self._model = self.model

    def save(self, model_name):
        self.model.save(model_name)


# agent to conduct the steps
class Agent:

    def __init__(self):
        self.D = deque()
        self.epsilon = INITIAL_EPSILON
        self.brain = Brain()
        self.modelCount = 0

    # do action
    def act(self, state):

        if random.random() <= self.epsilon:
            action_val = self.act_random()

        else:
            action_val = self.brain.predict_action(state)  # input a stack of 4 images, get the prediction

        return action_val

    def act_random(self):

        return random.randrange(5)

    def observe(self, state, action_value, reward, new_state, terminal_reached):

        if len(self.D) > REPLAY_SIZE:
            self.D.popleft()
        self.D.append((state, action_value, reward, new_state, terminal_reached))

    def replay(self):
        # sample a mini batch to train on
        mini_batch = random.sample(self.D, BATCH)
        self.brain.train(mini_batch)

    def update_brain(self):
        self.brain.update_target_model()

    def update_epsilon(self):
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / REPLAY_SIZE

    def save_brain(self):
        model_name = 'testModel' + str(self.modelCount) + '.h5'
        self.brain.save(model_name)
        self.modelCount += 1


# environment that run the training

class Environment:

    def __init__(self):
        self.agent = Agent()
        self.game = Game(ENV_LOCATION)

    def run(self):
        #  fill D
        #  do initial action to get initial state
        action = 0
        reward, state_t, terminal = self.game.perform_action(action, IMAGE_HEIGHT, IMAGE_WIDTH)
        for i in range(0, OBSERVER):
            action = self.agent.act_random()
            reward, state_t1, terminal = self.game.perform_action(action, IMAGE_HEIGHT, IMAGE_WIDTH)
            self.agent.observe(state_t, action, reward, state_t1, terminal)
            state_t = state_t1

        # reset agent
        self.game.reset()

        # train agent
        #  do initial action to get initial state
        action = 0
        reward, state_t, terminal = self.game.perform_action(action, IMAGE_HEIGHT, IMAGE_WIDTH)
        for i in range(0, TOTAL_EPI):
            action = self.agent.act(state_t)
            reward, state_t1, terminal = self.game.perform_action(action, IMAGE_HEIGHT, IMAGE_WIDTH)
            self.agent.observe(state_t, action, reward, state_t1, terminal)
            self.agent.replay()
            self.agent.update_epsilon()

            if i % C == 0:
                self.agent.update_brain()
                self.save_model()
            if terminal:
                action = 0
                reward, state_t1, terminal = self.game.perform_action(action, IMAGE_HEIGHT, IMAGE_WIDTH)

            state_t = state_t1

        self.game.close()

    def test(self, model_name):
        self.game = Game(ENV_LOCATION)
        # todo : fill method to test saved models
        # load model based on the model name : Agent -> brain
        # select action from loaded model brain : Agent
        # perform action to game
        # keep rewards (accumulate to termination)

    def save_model(self):
        self.agent.save_brain()


# -- Main -- #

environment = Environment()

try:
    environment.run()
finally:
    environment.save_model()

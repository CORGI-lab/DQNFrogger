import numpy as np
from collections import deque
import random

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import load_model

from GameHandler import Game

# Parameters
IMAGE_HEIGTH = 100
IMAGE_WIDTH = 100
ENV_LOCATION = "build/froggerNew"
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
OBSERVER = 50000  # filling D (experience replay data)
REPLAY_SIZE = 100000  # size of D

#
TOTAL_EPI = 9000000
C = 10000  # update q`


# -- Brain --#
class Brain:

    def __init__(self):
        self.model = self._createModel()   # q model
        self._model = self._createModel()  # q` model ( used to calculate predication for error)

    # no init of network param : we can use normal dist.-:init=lambda shape, name: normal(shape, scale=0.01, name=name),
    def _createModel(self):

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(IMAGE_HEIGTH, IMAGE_WIDTH, STACK_SIZE)))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('linear'))
        #model.add(Activation('relu'))
        model.add(Dense(5))

        opt = RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=0.01)
        #opt = Adam(lr=LEARNING_RATE)
        model.compile(loss='mean_squared_error', optimizer=opt)  # mse for dqn

        return model

    # predict action from state images
    def predictAction(self, state):

        q = self.model.predict(state)  # input a stack of 4 images, get the prediction
        max_Q = np.argmax(q)
        actionVal = max_Q

        return actionVal

    # train model using the re play queue
    def train(self, minibatch):

        self.trainingLoss = 0

        inputs = np.zeros((BATCH, IMAGE_HEIGTH, IMAGE_WIDTH, STACK_SIZE))  # 2500, 100, 100, 4
        targets = np.zeros((inputs.shape[0], 5))  # 2500, 4
        targets_ = np.zeros((inputs.shape[0], 5))  # 2500, 4

        # Now we do the experience replay
        for j in range(0, len(minibatch)):
            state_t = minibatch[j][0]  # state
            action_t = minibatch[j][1]  # action
            reward_t = minibatch[j][2]  # reward
            state_t1 = minibatch[j][3]  # new state
            terminal = minibatch[j][4]  # is terminal reached or not

            inputs[j:j + 1] = state_t  # saved down s_t as input

            targets[j] = self.model.predict(state_t)  # predict from the model each action value
            targets_[j] = self.model.predict(state_t1)  # predict from the online model each action value
            Q_sa = self._model.predict(state_t1)  # predict to get arg max Q to cal TD

            if terminal:
                targets[j, action_t] = reward_t  # if terminal only set target as reward for the action
            else:
                targets[j, action_t] = reward_t + GAMMA * Q_sa[0][np.argmax(targets_[j])]

        self.trainingLoss += self.model.train_on_batch(inputs, targets)

    # update q`
    def updateTargetModel(self):
        self._model = self.model

    def save(self, modelName):
        self.model.save(modelName)


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
            actionVal = self.actRandom()

        else:
            actionVal = self.brain.predictAction(state)  # input a stack of 4 images, get the prediction

        return actionVal

    def actRandom(self):

        return random.randrange(5)

    def observe(self, state, actionValue, reward, newState, terminalReached):

        if len(self.D) > REPLAY_SIZE:
            self.D.popleft()
        self.D.append((state, actionValue, reward, newState, terminalReached))

    def replay(self):
        # sample a minibatch to train on
        minibatch = random.sample(self.D, BATCH)
        self.brain.train(minibatch)

    def updateBrain(self):
        self.brain.updateTargetModel()

    def updateEpsiolon(self):
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / REPLAY_SIZE

    def saveBrain(self):
        modelName = 'testModel' + str(self.modelCount) + '.h5'
        self.brain.save(modelName)
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
        reward, state_t, terminal = self.game.perform_action(action, IMAGE_HEIGTH, IMAGE_WIDTH)
        for i in range(0, OBSERVER):
            action = self.agent.actRandom()
            reward, state_t1, terminal = self.game.perform_action(action, IMAGE_HEIGTH, IMAGE_WIDTH)
            self.agent.observe(state_t, action, reward, state_t1, terminal)
            state_t = state_t1

        # reset agent
        self.game.reset()

        # train agent
        #  do initial action to get initial state
        action = 0
        reward, state_t, terminal = self.game.perform_action(action, IMAGE_HEIGTH, IMAGE_WIDTH)
        for i in range(0, TOTAL_EPI):
            action = self.agent.act(state_t)
            reward, state_t1, terminal = self.game.perform_action(action, IMAGE_HEIGTH, IMAGE_WIDTH)
            self.agent.observe(state_t, action, reward, state_t1, terminal)
            self.agent.replay()
            self.agent.updateEpsiolon()

            if i % C == 0:
                self.agent.updateBrain()
                self.saveModel()
            if terminal:
                action = 0
                reward, state_t1, terminal = self.game.perform_action(action, IMAGE_HEIGTH, IMAGE_WIDTH)

            state_t = state_t1

        self.game.close()

    def test(self, model_name):
        self.game = Game(ENV_LOCATION)
        # todo : fill method to test saved models
        # load model based on the model name : Agent -> brain
        # select action from loaded model brain : Agent
        # perform action to game
        # keep rewards (cumilate to teminatition)

    def saveModel(self):
        self.agent.saveBrain()


# -- Main -- #

environment = Environment()

try:
    environment.run()
finally:
    environment.saveModel()

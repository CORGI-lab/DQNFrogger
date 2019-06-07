import skimage
from mlagents.envs import UnityEnvironment
import numpy as np
from collections import deque
import random

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD, Adam, RMSprop

# Parameters
IMAGE_HEIGTH = 100
IMAGE_WIDTH = 100
ENV_LOCATION = "build/froggerNew"

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


# -- game handling class -- #
class Game:

    # set up unity ml agent environment

    def __init__(self):
        self.loadEnv(0)

    def loadEnv(self, wid):
        # load env
        env_name = ENV_LOCATION
        self.env = UnityEnvironment(env_name, worker_id=wid)
        # Set the default brain to work with
        self.default_brain = self.env.brain_names[0]
        self.brain = self.env.brains[self.default_brain]
        # Reset the environment - train mode enabled
        env_info = self.env.reset(train_mode=True)[self.default_brain]

    # this frogger game action space is 5, actions[0] = selected action (action = [[1]])
    # actions
    # 1 - up, 2 - down , 3- left , 4 -right , 0 - do nothing

    def performAction(self, actionValue, numberOfFrames=STACK_SIZE):
        action = [[0]]
        action[0] = actionValue
        terminal = False  # indication of terminal state
        size = (IMAGE_HEIGTH, IMAGE_WIDTH, numberOfFrames)  # create list to keep frames
        stack = np.zeros(size)
        reward = 0  # rewards for all the frames

        # first frame after action
        env_info = self.env.step(action)[self.default_brain]  # send action to brain
        reward = round(env_info.rewards[0], 5)  # get reward
        newState = env_info.visual_observations[0][0]  # get state visual observation
        newStateGray = skimage.color.rgb2gray(newState)  # covert to gray scale
        newStateGray = skimage.transform.resize(newStateGray, (IMAGE_HEIGTH, IMAGE_WIDTH))
        # check terminal reached
        if reward == -1 or reward == -2:
            terminal = True

        # add the state to the 0 th position
        stack[:, :, 0] = newStateGray

        # get stack of frames after the action
        for i in range(1, numberOfFrames):
            env_info = self.env.step()[self.default_brain]  # change environment to next step without action
            st = env_info.visual_observations[0][0]
            stGray = skimage.color.rgb2gray(st)
            stGray = skimage.transform.resize(stGray, (IMAGE_HEIGTH, IMAGE_WIDTH))
            stack[:, :, i] = stGray
            # if terminal only consider the reward for terminal
            if env_info.rewards[0] == -1 or env_info.rewards[0] == -2:
                terminal = True
                reward = round(env_info.rewards[0], 5)
            elif not terminal:
                # if it got a positive reward for move up let it have it
                if reward < 0:
                    reward = round(env_info.rewards[0], 5)  # get reward

        # reshape for Keras
        # noinspection PyArgumentList
        stack = stack.reshape(1, stack.shape[0], stack.shape[1], stack.shape[2])  # 1*100*100*4

        return reward, stack, terminal

    # close environment
    def close(self):
        self.env.close()

    def reset(self):
        self.close()
        self.loadEnv(0)


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

        # Now we do the experience replay
        for j in range(0, len(minibatch)):
            state_t = minibatch[j][0]  # state
            action_t = minibatch[j][1]  # action
            reward_t = minibatch[j][2]  # reward
            state_t1 = minibatch[j][3]  # new state
            terminal = minibatch[j][4]  # is terminal reached or not

            inputs[j:j + 1] = state_t  # saved down s_t as input

            targets[j] = self.model.predict(state_t)  # predict from the model each action value
            Q_sa = self._model.predict(state_t1)  # predict to get arg max Q to cal TD

            if terminal:
                targets[j, action_t] = reward_t  # if terminal only set target as reward for the action
            else:
                targets[j, action_t] = reward_t + GAMMA * Q_sa[np.argmax(targets[j])]

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
        self.game = Game()

    def run(self):
        #  fill D
        #  do initial action to get initial state
        action = 0
        reward, state_t, terminal = self.game.performAction(action)
        for i in range(0, OBSERVER):
            action = self.agent.actRandom()
            reward, state_t1, terminal = self.game.performAction(action)
            self.agent.observe(state_t, action, reward, state_t1, terminal)
            state_t = state_t1

        # reset agent
        self.game.reset()

        # train agent
        #  do initial action to get initial state
        action = 0
        reward, state_t, terminal = self.game.performAction(action)
        for i in range(0, TOTAL_EPI):
            action = self.agent.act(state_t)
            reward, state_t1, terminal = self.game.performAction(action)
            self.agent.observe(state_t, action, reward, state_t1, terminal)
            self.agent.replay()
            self.agent.updateEpsiolon()

            if i % C == 0:
                self.agent.updateBrain()
                self.saveModel()
            if terminal:
                action = 0
                reward, state_t1, terminal = self.game.performAction(action)

            state_t = state_t1

        self.game.close()

    def saveModel(self):
        self.agent.saveBrain()


# -- Main -- #

environment = Environment()

try:
    environment.run()
finally:
    environment.saveModel()

import skimage
from mlagents.envs import UnityEnvironment
import numpy as np


# -- game handling class -- #
class Game:

    # set up unity ml agent environment

    def __init__(self, game_location):
        self.ENV_LOCATION = game_location
        self.load_env(0)

    def load_env(self, wid):
        # load env
        env_name = self.ENV_LOCATION
        self.env = UnityEnvironment(env_name, worker_id=wid)
        # Set the default brain to work with
        self.default_brain = self.env.brain_names[0]
        self.brain = self.env.brains[self.default_brain]
        # Reset the environment - train mode enabled
        env_info = self.env.reset(train_mode=True)[self.default_brain]

    # this frogger game action space is 5, actions[0] = selected action (action = [[1]])
    # actions
    # 1 - up, 2 - down , 3- left , 4 -right , 0 - do nothing

    def perform_action(self, action_value, image_height, image_width, number_of_frames=4):
        action = [[0]]
        action[0] = action_value
        terminal = False  # indication of terminal state
        size = (image_height, image_width, number_of_frames)  # create list to keep frames
        stack = np.zeros(size)

        # first frame after action
        env_info = self.env.step(action)[self.default_brain]  # send action to brain
        reward = round(env_info.rewards[0], 5)  # get reward
        new_state = env_info.visual_observations[0][0]  # get state visual observation
        new_state_gray = skimage.color.rgb2gray(new_state)  # covert to gray scale
        new_state_gray = skimage.transform.resize(new_state_gray, (image_height, image_width))
        # check terminal reached
        if env_info.local_done:
            terminal = True

        # add the state to the 0 th position of stack
        stack[:, :, 0] = new_state_gray

        # get stack of frames after the action
        for i in range(1, number_of_frames):
            env_info = self.env.step()[self.default_brain]  # change environment to next step without action
            st = env_info.visual_observations[0][0]
            st_gray = skimage.color.rgb2gray(st)
            st_gray = skimage.transform.resize(st_gray, (image_height, image_width))
            stack[:, :, i] = st_gray
            # if terminal only consider the reward for terminal
            if env_info.local_done:
                terminal = True
                reward = round(env_info.rewards[0], 5)
            elif not terminal:
                # if it got a positive reward for move up let it have it
                if reward < 0:
                    reward = round(env_info.rewards[0], 5)  # get reward

        # reshape for Keras
        # noinspection PyArgumentList
        stack = stack.reshape(1, stack.shape[0], stack.shape[1], stack.shape[2])

        return reward, stack, terminal

    # close environment
    def close(self):
        self.env.close()

    def reset(self):
        self.close()
        self.load_env(0)

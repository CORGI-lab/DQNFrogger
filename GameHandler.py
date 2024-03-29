import skimage
from mlagents.envs import UnityEnvironment
import numpy as np


# -- game handling class -- #
class Game:

    """
    set up unity ml agent environment
    @:param game_location  : file path for executable
    """
    def __init__(self, game_location):
        self.ENV_LOCATION = game_location
        self.load_env(0)

    """
    load unity environment
    @:param wid  : id for the worker in unity environment 
    """
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
    """
    performs a given action to the unity game 
    @:param action_value : action to be execute
    @:param image_height : Desire image height 
    @:param image_width  : Desire image width 
    @:param number_of_frames : stack size (this number of frames will e stack together by performing no op action )
    @:return reward : reward for the action 
    @:return stack  : stack of frames
    @:return terminal : if game reached terminal state or not
    """
    def perform_action(self, action_value, image_height, image_width, number_of_frames=4):
        action = [[0]]
        action[0] = action_value
        terminal = False  # indication of terminal state
        # 3 - R, G, B
        size = (image_height, image_width, 3, number_of_frames)  # create list to keep frames
        stack = np.zeros(size)

        # first frame after action
        env_info = self.env.step(action)[self.default_brain]  # send action to brain
        reward = round(env_info.rewards[0], 5)  # get reward
        new_state = env_info.visual_observations[0][0]  # get state visual observation
        # new_state_gray = skimage.color.rgb2gray(new_state)  # covert to gray scale
        new_state_gray = skimage.transform.resize(new_state, (image_height, image_width))  # resize
        # check terminal reached
        if env_info.local_done[0]:
            terminal = True

        # add the state to the 0 th position of stack
        stack[:, :, :, 0] = new_state_gray

        # get stack of frames after the action
        for i in range(1, number_of_frames):
            env_info = self.env.step()[self.default_brain]  # change environment to next step without action
            st = env_info.visual_observations[0][0]
            #st_gray = skimage.color.rgb2gray(st)
            st_gray = skimage.transform.resize(st, (image_height, image_width))
            stack[:, :, :, i] = st_gray
            # if terminal only consider the reward for terminal
            if env_info.local_done[0]:
                terminal = True
                reward = round(env_info.rewards[0], 5)

        # reshape for Keras
        # noinspection PyArgumentList
        stack = stack.reshape(1, stack.shape[0], stack.shape[1], stack.shape[2], stack.shape[3])

        return reward, stack, terminal

    """
    close environment
    """
    def close(self):
        self.env.close()

    """
    Reset environment 
    """
    def reset(self):
        self.close()
        self.load_env(0)

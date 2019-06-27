import skimage
from mlagents.envs import UnityEnvironment

# TODO : finish
ENV_LOCATION = "build/froggerNew"

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

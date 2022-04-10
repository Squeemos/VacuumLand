from typing import Any, Dict, Optional, Type, Union, Tuple, List

import gym
import numpy as np

class VacuumLand(gym.Env):
    def __init__(self,
                    height:Type[int]=5,
                    width:Type[int]=5,
                    trash:Type[int]=5,
                    as_image:Type[bool]=False,
                    penalty:Optional[Union[bool,float,int]]=True,
                    max_steps:Optional[Union[bool,int]]=None,
                    seed:Optional[int]=None):
        """
            height    : Height of the board (Default: 5)
            width     : Width of the board (Default: 5)
            trash     : How many pieces of trash to be placed on the board (must be less than height * width) (Default: 5)
            as_image  : Whether to return the board as an image with dimensions (height, width, 1) or as a 2D array with dimensions (height, width) (Default: False)
            pentaly   : Penalty applied to stepping onto location without trash.
                        True      : Penalty is -.01
                        False     : Penatly is 0
                        Int/Float : Penatly is custom value
                        (Default: True, -.01)

            max_steps : How many steps to take in the environment.
                        None : Height * Width
                        Int  : Custom value
                        (Default: None, Height * Width)

            seed      : Value to seed np.random with
                        None : Random seed
                        Int  : np.random.seed(seed)
                        (Default: None, Random seed)

            TODO : Handle environment with changing starting position
            TODO : Handle having obstacles/walls in the environment
            TODO : Handle having a different form of reward (other than just 1 / trash)
            TODO : Maybe add in the ability to take diagonal steps?
        """
        super(VacuumLand,self).__init__()
        self.height = height
        self.width = width
        self.trash = trash
        # Make sure there isn't more trash than spaces
        assert self.trash < self.height * self.width, f"Trash must be less than height * width: {self.height * self.width}, trash amount provdided: {self.trash}"

        # Used if the agnet accepts image argument
        # Will return the board with shape (height,width,1)
        self.as_image = as_image
        self.internal_shape = (self.height,self.width)

        # Setting up spaces
        if self.as_image:
            self.observation_shape = (self.height,self.width,1)
            self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = self.observation_shape, dtype = np.uint8)
            self.player_val = 122
            self.trash_val = 255
        else:
            self.observation_shape = (self.height,self.width)
            self.observation_space = gym.spaces.Box(low = 0, high = 2, shape = self.observation_shape, dtype = np.uint8)
            self.player_val = 1
            self.trash_val = 2

        # Up,Down,Left,Right
        self.action_space = gym.spaces.Discrete(4,)

        # Reward for collecting the trash
        self.reward_amount = 1 / self.trash

        # Penalty for moving to a space where there isn't trash
        if isinstance(penalty,bool):
            self.penalty = -.01 if penalty else 0
        elif isinstance(penalty,float) or isinstance(penalty,int):
            self.penalty = penalty
        else:
            assert isinstance(penalty,bool) or isinstance(penalty,float) or isinstance(penalty,int), f"Penalty must be one of type {[bool,float,int]}"


        # Max steps involves stepping over every place in the board (or custom amount)
        if isinstance(max_steps,int):
            assert max_steps > 0, f"max_steps must be greater than 0. max_steps provided {max_steps}"
            self.max_steps = max_steps
        elif max_steps is not None:
            assert isinstance(max_steps,int), f"max_steps must be of type int. Type provided: {type(max_steps)}"
        else:
            self.max_steps = self.width * self.height

        # Used for creating the random state of the board
        self.seed = seed

    def reset(self) -> Type[np.ndarray]:
        """
            Resets the board and the starting position. Agent always starts at (0,0) without the possibility of starting in position with trash.
            Also seeds the randomization.

            TODO: Handle environment with changing starting position
        """
        # Seed the shuffle if there's a seed, otherwise it's random
        if self.seed is not None:
            np.random.seed(self.seed)
        # Starting agent position
        self.agent_pos = (0,0)

        # Create a mask of rewards, leaving a spot for (0,0) to have a 0
        mask = np.zeros(self.height * self.width - 1,dtype=np.uint8)
        # Trash has the value 2
        mask[:self.trash] = self.trash_val
        np.random.shuffle(mask)
        # Create the board
        self.board = np.concatenate((np.full(1,self.player_val,dtype=np.uint8),mask)).reshape(self.internal_shape)

        # Reset number of steps taken
        self.steps = 0

        if self.as_image:
            return self.board[...,np.newaxis]
        else:
            return self.board

    def step(self,action:Type[int]) -> Tuple[Type[np.ndarray], Type[float], Type[bool], Dict]:
        """
            Take a step in the environment with the chosen action. Currently handles Up/Down/Left/Right

            action : Which action to take

            TODO : Maybe add in the ability to take diagonal steps?
        """
        # Make sure the action can be taken
        assert action < self.action_space.n, f"Action attempted {action} in a space with only {self.action_space.n} possible actions"
        assert self.steps <= self.max_steps, f"Trying to step in a done environment."

        prev_location = self.agent_pos
        # Up
        if action == 0:
            if self.agent_pos[0] != 0:
                self.agent_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        # Down
        elif action == 1:
            if self.agent_pos[0] != self.height - 1:
                self.agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        # Left
        elif action == 2:
            if self.agent_pos[1] != 0:
                self.agent_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        # Right
        elif action == 3:
            if self.agent_pos[1] != self.width - 1:
                self.agent_pos = (self.agent_pos[0], self.agent_pos[1] + 1)

        # If the agent didn't move
        if self.agent_pos == prev_location:
            reward = self.penalty
        else:
            reward = self.board[self.agent_pos]
            if reward == self.trash_val:
                reward = self.reward_amount
            self.board[self.agent_pos] = 1
            self.board[prev_location] = 0

        # Take a step
        self.steps += 1
        # If the environment is done after this step
        if self.steps == self.max_steps:
            done = True
        else:
            # Check if there's still a reward
            if self.trash_val in self.board:
                done = False
            else:
                done = True

        if self.as_image:
            return self.board[...,np.newaxis], reward, done, {}
        else:
            return self.board, reward, done, {}

    def render(self, mode=None) -> None:
        """
            Renders the environment to the console with a print statement.

            TODO : Render the environment using PyGame or opencv for more interesting visual
            TODO : Handle the rendering modes
        """
        try:
            print(self.board)
        except AttributeError:
            print("Trying to render an environment which has not been reset. Please call 'env.reset()' before rendering")

    def seed(self, seed=None) -> List[Union[None,int]]:
        """
            Overwritten to comply with gym's environment requirement

            TODO : Make this actually do something, as of right now it just doesn't do anything
        """
        # if there is no seed, return an empty list
        if seed is None:
            return []
        # return the list of seeds used by RNG(s) in the environment
        return [seed]

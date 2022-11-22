from typing import Any, Optional, Union

import gym
import numpy as np
import warnings
from enum import Enum

class ActionType(Enum):
    CARDINAL = 4
    DIAGONAL = 8

    @property
    def action_names(self):
        return {
            0 : "Up",
            1 : "Down",
            2 : "Left",
            3 : "Right",
            4 : "Up Left",
            5 : "Up Right",
            6 : "Down Left",
            7 : "Down Right",
        }

class VacuumLand(gym.Env):
    '''
        VacuumLand Class for an environment where a robot picks up trash
    '''
    metadata = {"render_modes" : ["human", "rgb_array", "better_array"]}

    def __init__(self,
                    height : int = 5,
                    width : int = 5,
                    trash : int = 5,
                    as_image : bool = False,
                    penalty : Optional[Union[bool, float, int]] = True,
                    max_steps : Optional[int] = None,
                    reward : Optional[Union[int, float]] = None,
                    starting_pos : Optional[Union[tuple[int, int], list[int, int], str]] = None,
                    action_type : Optional[ActionType] = ActionType.CARDINAL,
                    seed : Optional[int] = None):
        """
            height    : Height of the board
                            (Default: 5)

            width     : Width of the board
                            (Default: 5)

            trash     : How many pieces of trash to be placed on the board (must be less than height * width)
                            (Default: 5)

            as_image  : Whether to return the board as an image with dimensions (height, width, 1) or as a 2D array with dimensions (height, width)
                            (Default: False)

            pentaly   : Penalty applied to stepping onto location without trash
                            True      : -.01
                            False     : 0
                            Int/Float : Custom value
                            (Default: True, -.01)

            max_steps : How many steps to take in the environment
                            None : Height * Width
                            Int  : Custom value
                            (Default: None, Height * Width)

            reward    : How much reward to give each time the robot collects a piece of trash
                            None      : 1 / trash
                            Int/Float : Custom value
                            (Default: None, 1 / trash)

            action_type : The type of action the agent will be allowed to take
                            ActionType.CARDINAL : Cardinal movement direction (4)
                            ActionType.DIAGONAL : Diagonal movement directions (8)
                            (Default: ActionType.CARDINAL)

            starting_pos : The location the agent stats out at
                            Tuple[Int, Int] : x, y for the agent to start at
                            List[Int, Int] : x, y for the agent to start at
                            str : What type of starting position (random)
                            (Default: None, (0, 0))

            seed      : Value to seed np.random with
                            None : Random seed
                            Int  : np.random.seed(seed)
                            (Default: None, Random seed)

            TODO : Handle having obstacles/walls in the environment
            TODO : Maybe add in the ability to take diagonal steps?
        """
        super(VacuumLand, self).__init__()

        # Check types for height, width, and trash
        assert isinstance(height, int) and not isinstance(height, bool), f"height should be of type int rather than type {type(height)}"
        assert isinstance(width, int) and not isinstance(width, bool), f"height should be of type int rather than type {type(width)}"
        assert isinstance(trash, int) and not isinstance(trash, bool), f"height should be of type int rather than type {type(trash)}"

        assert height > 0, f"height should be > 0; passed value: {height}"
        assert width > 0, f"width should be > 0; passed value: {width}"
        assert trash > 0, f"trash should be > 0; passed value: {trash}"

        self.height = height
        self.width = width
        self.trash = trash
        # Make sure there isn't more trash than spaces
        assert self.trash < self.height * self.width, f"Trash must be less than height * width: {self.height * self.width}, trash amount provdided: {self.trash}"

        # Used if the agnet accepts image argument
        # Will return the board with shape (height, width, 1)
        assert isinstance(as_image, bool), f"as_image should be of type int rather than type {type(as_image)}"
        self.as_image = as_image
        self.internal_shape = (self.height, self.width)

        # Setting up spaces
        if self.as_image:
            self.observation_shape = (self.height, self.width, 1)
            self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = self.observation_shape, dtype = np.uint8)
            self.player_val = 122
            self.trash_val = 255
        else:
            self.observation_shape = (self.height, self.width)
            self.observation_space = gym.spaces.Box(low = 0, high = 2, shape = self.observation_shape, dtype = np.uint8)
            self.player_val = 1
            self.trash_val = 2

        # Up,Down,Left,Right
        self.action_type = action_type
        self.action_space = gym.spaces.Discrete(action_type.value,)

        # Reward for collecting the trash
        if reward is not None:
            assert isinstance(reward, (int, float)) and not isinstance(reward, bool), f"Reward amount should be one of type {[int, float]} instead of {type(reward)}"
            self.reward_amount = reward
        else:
            self.reward_amount = 1 / self.trash

        # Penalty for moving to a space where there isn't trash
        if isinstance(penalty, bool):
            self.penalty = -.01 if penalty else 0
        elif isinstance(penalty, (float, int)):
            self.penalty = penalty
        else:
            assert isinstance(penalty, (bool, float, int)), f"Penalty must be one of type {[bool, float, int]}"

        # Max steps involves stepping over every place in the board (or custom amount)
        if isinstance(max_steps, int):
            assert max_steps > 0, f"max_steps must be greater than 0. max_steps provided {max_steps}"
            self.max_steps = max_steps
        elif max_steps is not None:
            assert isinstance(max_steps, int), f"max_steps must be of type int. Type provided: {type(max_steps)}"
        else:
            self.max_steps = self.width * self.height

        # Used for creating the random state of the board
        if seed is not None:
            assert isinstance(seed, int) and not isinstance(seed, bool), f"seed must be of type int rather than type {type(seed)}"
            self._seed = seed

        # Setting the reward range
        self.reward_range = (self.max_steps * self.penalty), 1
        self.steps = -1

        # Set the starting position
        if starting_pos is None:
            starting_pos = (0, 0)
        elif isinstance(starting_pos, (tuple, list)):
            assert len(starting_pos) == 2, f"Starting position must be x, y pair. Not {len(starting_pos)}"
            for item in starting_pos:
                assert isinstance(item, int), f"Starting positions should be of type int rather than type {type(item)}"
        elif isinstance(starting_pos, str):
            pass
        else:
            assert isinstance(starting_pos, (list, tuple, str)), f"Starting positions should be one of type {[list, tuple, str]}"
        self.starting_pos = starting_pos

    def reset(self, seed : Optional[int] = None, return_info : bool = False) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
        """
            Resets the board and the starting position. Agent always starts at (0,0) without the possibility of starting in position with trash.
            Also seeds the randomization.

            seed        : Optional int to redo the seeding with
            return_info : Whether to return additional info related to the environment
        """
        if seed is not None:
            assert isinstance(seed, int), f"Seed must be of type int, passed {type(seed)}"
            self._seed = seed
            np.random.seed(self._seed)
        # Starting agent position
        if isinstance(self.starting_pos, (tuple, list)):
            self.agent_pos = self.starting_pos
        elif isinstance(self.starting_pos, str):
            if self.starting_pos == "random":
                x = np.random.randint(self.width)
                y = np.random.randint(self.height)
                self.agent_pos = (x, y)
            else:
                warnings.warn(
                    f"Unhandled starting position type: {self.starting_pos} "
                    f"Currently handled tpyes: random "
                    "Setting position to (0, 0) "
                )
                self.agent_pos = (0, 0)

        # Create a mask of rewards, leaving a spot for agent position
        mask = np.zeros(self.height * self.width - 1, dtype = np.uint8)
        # Set trash values
        mask[:self.trash] = self.trash_val
        # Randomly shuffle the trash
        np.random.shuffle(mask)

        # Create the board
        left_size = self.agent_pos[0] * self.width + self.agent_pos[1]
        left = mask[:left_size]
        right = mask[left_size:]
        player = np.full(1, self.player_val, dtype = np.uint8)
        # Stuff on both sides of agent
        if len(left) and len(right):
            self.board = np.concatenate((left, player, right)).reshape(self.internal_shape)
        # Only things to the left of agent (bottom right corner)
        elif len(left) and not len(right):
            self.board = np.concatenate((left, player)).reshape(self.internal_shape)
        # Only things to right of agent (top left corner)
        elif not len(left) and len(right):
            self.board = np.concatenate((player, right)).reshape(self.internal_shape)

        # Reset number of steps taken
        self.steps = 0

        # Reset the number of trash
        self.current_trash = self.trash

        if return_info:
            if self.as_image:
                return self.board[..., np.newaxis], {}
            else:
                return self.board, {}
        else:
            if self.as_image:
                return self.board[..., np.newaxis]
            else:
                return self.board

    def step(self, action : int) -> tuple[np.ndarray, float, bool, dict]:
        """
            Take a step in the environment with the chosen action. Currently handles Up/Down/Left/Right

            action : Which action to take
        """
        # Make sure the action can be taken
        assert action >= 0 and action < self.action_space.n, f"Action attempted {action} in an action space [0, {self.action_space.n})"
        if self.steps != -1:
            assert self.steps <= self.max_steps, f"Trying to step in a done environment."
        else:
            raise Exception(f"Trying to step in an environment that has not been reset. Please call .reset() before .step()")

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

        # Diagonal Left Up
        elif action == 4:
            if self.agent_pos[0] != 0 and self.agent_pos[1] != 0:
                self.agent_pos = (self.agent_pos[0] - 1, self.agent_pos[1] - 1)

        # Diagonal Right Up
        elif action == 5:
            if self.agent_pos[0] != 0 and self.agent_pos[1] != self.width - 1:
                self.agent_pos = (self.agent_pos[0] - 1, self.agent_pos[1] + 1)

        # Diagoanl Down Left
        elif action == 6:
            if self.agent_pos[0] != self.height - 1 and self.agent_pos[1] != 0:
                self.agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1] - 1)

        # Digoanl Down Right
        elif action == 7:
            if self.agent_pos[0] != self.height - 1 and self.agent_pos[1] != self.width - 1:
                self.agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1] + 1)

        # If the agent didn't move
        if self.agent_pos == prev_location:
            reward = self.penalty
        else:
            # Agent moved, get the value at location
            reward = self.board[self.agent_pos]
            if reward == self.trash_val:
                # If we collected trash, reward
                reward = self.reward_amount
                self.current_trash -= 1
            else:
                # Did not collect trash, penalty
                reward = self.penalty
            self.board[self.agent_pos] = self.player_val
            self.board[prev_location] = 0

        # Take a step
        self.steps += 1
        # If the environment is done after this step
        if self.steps == self.max_steps or not self.current_trash:
            done = True
            self.steps = -1
        else:
            done = False

        if self.as_image:
            return self.board[..., np.newaxis], reward, done, {}
        else:
            return self.board, reward, done, {}

    def render(self, mode : str = "human") -> Union[None, np.ndarray]:
        """
            Renders the environment to the console with a print statement.

            TODO : Render the environment using PyGame or opencv for more interesting visual
            TODO : Handle the rendering modes
            TODO : Handle rendering the environment to render slower to be readable
        """
        if hasattr(self, 'board'):
            if mode == "human":
                print(self.board)
            elif mode == "rgb_array":
                display_board = np.stack((self.board, self.board, self.board), axis=2)
                display_board[self.agent_pos] = [255, 0, 0]
                return display_board
            elif mode == "better_array":
                for row in self.board:
                    print(row)
            else:
                super(VacuumLand, self).render(mode = mode)
        else:
            print("Trying to render an environment which has not been reset. Please call 'env.reset()' before rendering")

    def seed(self, seed : int = None) -> list[Union[None, int]]:
        """
            Seeds the environment with the provided seed
        """
        # if there is no seed, return an empty list
        if seed is None:
            return []

        assert isinstance(seed, int), f"seed must be an int or None, passed {type(seed)}"
        # Set the seed
        self._seed = seed

        # Warn for deprecation
        warnings.warn(
            "Function `env.seed(seed)` is marked as deprecated and will be removed in the future. "
            "Please use `env.reset(seed=seed) instead."
        )
        # return the list of seeds used by RNG(s) in the environment
        return [seed]

    def close(self) -> None:
        '''
            Cleanup anything allocated or opened
        '''

        pass

    def prompt_human_move(self) -> int:
        '''
            Prompts a human player to make a move
        '''
        print_string = "Which action would you like to take?\n"
        for n in range(self.action_type.value):
            print_string += f"{n:2}: {self.action_type.action_names[n]}\n"

        action = int(input(print_string))

        return action

    @classmethod
    def register(cls, *args, **kwargs) -> None:
        '''
            Registers a VacuumLand environment with a list of kwargs
        '''
        gym.envs.registration.register(
            id = "VacuumLand-v0",
            entry_point = VacuumLand,
            max_episode_steps = kwargs.get("max_steps", 400),
            kwargs = kwargs,
        )

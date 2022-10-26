# VaccumLand
Gym environment for a trash finding robot

Current supported versions: Python 3.8x

Required Packages: gym

# Register Environment With gym
Sample for registering a VacuumLand env with a 20x20 board with 10 trash, a penalty, 400 max steps, and cardinal actions
```python
from vacuum_land import VacuumLand, ActionType
import gym

env_kwargs = {
  'height' : 20, 'width' : 20, 'trash' : 10,
  'as_image' : False, 'penalty' : True, 'seed' : 0,
  'max_steps' : 400, "action_type" : ActionType.CARDINAL,
}
VacuumLand.register(env_kwargs)
```

# Environment Creation
```python
from vacuum_land import VacuumLand, ActionType

env = VacuumLand(height=5, width=5, trash=5, as_image=False, penalty=True, max_steps=25, seed=0)
# VacuumLand.register()
# env = gym.make("VacuumLand-v0")

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    env.render()
```

# Current Features
- height
    - Height of the grid
    - default = 5
- width
    - Width of the grid
    - default = 5
- trash
    - How many pieces of trash to randomly place in the grid
    - default = 5
- as_image
    - When True: Returns the observation as (Channel, Height, Width) in PyTorch image format
    - When False: Returns the observation as (Height, Width)
    - default = False
- penalty
    - The penalty for taking a step in the environment that doesn't have a piece of trash on it.
    - When True: Penalty for taking entering a space without trash is -0.01
    - When False: No penalty is applied for taking a step
    - When Numerical Type: Custom value for penalty (Ex. -0.5)
    - default = True
- max_steps
    - The maximum number of steps for the environment
    - When None: Defaults to Height * Width
    - When Int: Custom defined number for max steps (Ex. 400)
    - default = None (Height * Width)
- reward
    - The reward to return when the robot steps over a piece of trash
    - When None: Defaults to 1 / trash
    - When Numerical Type: Custom value for reward (Ex. 1)
- seed
    - The seed for environment creation/reset
    - When None: Randomly seeds the environment when calling np.random.shuffle()
    - When Int: Custom value for seed, gets called during reset in np.random.seed(seed)
    - default = None
- action_type
    - The type of actions you want the environment to use
    - When ActionyType.CARDINAL: Cardinal directions for movement
    - When ActionType.DIAGONAL: Cardinal and diagonal directions for movement
    - default = ActionType.CARDINAL


# TODOs
- [ ] Get code to follow PEP8 standard
- [ ] Convert to a PyPi package to install instead of including the file
- [ ] Make it check which version of Python the virtual environment is using, so it can adapt to using different version of Python
- [ ] Register the environment on inclusion of the file
- [ ] Handle environment with changing starting positions
- [ ] Handle having obstacles in the environment (such as holes and walls)
- [x] Have different style of reward (more than just 1 / total trash)
- [x] Multiple styles of action (Cardinal, Diagonal, etc)
- [ ] Better environment rendering (PyGame or OpenCV as well as text based)
- [ ] Ability to render the environment with a frame rate
- [x] Ability to render the environment for a human to play
- [ ] Ability to play a game as human in a better rendered environment
- [x] Make the seed function properly seed the environment (deprecated and handled inside reset)
- [ ] Ability to set custom boards
- [ ] Modifiers to player/trash value (instead of 1/2 for as_image=False and 122/255 for as_image=True)
- [x] Make a wrapper for registering the env that allows for kwargs

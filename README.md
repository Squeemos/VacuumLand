# VaccumLand
Gym environment for a trash finding robot

Current supported versions: Python 3.8x

# Environment Creation
```python
from vacuum_land import VacuumLand

env = VacuumLand(height=5, width=5, trash=5, as_image=False, penaly=True, max_steps=25, seed=0)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    env.render()
```

# Register Environment With gym
Sample for registering a VacuumLand env with a 20x20 board with 10 trash, a penalty, and 400 max steps
```python
from vacuum_land import VacuumLand
import gym

gym.envs.registration.register(
    id = "VacuumLand-v0",
    entry_point = VacuumLand,
    max_episode_steps = 400,
    kwargs = {'height' : 20, 'width' : 20, 'trash' : 10, 'as_image' : False, 'penalty' : True, 'seed' : 0, 'max_steps' : 400}
)
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
    - default = None
- seed
    - The seed for environment creation/reset
    - When None: Randomly seeds the environment when calling np.random.shuffle()
    - When Int: Custom value for seed, gets called during reset in np.random.seed(seed)
    - default = None


# TODOs
- [ ] Get code to follow PEP8 standard
- [ ] Convert to a PyPi package to install instead of including the file
- [ ] Make it check which version of Python the virtual environment is using, so it can adapt to using different version of Python
- [ ] Register the environment on inclusion of the file
- [ ] Handle environment with changing starting positions
- [ ] Handle having obstacles in the environment (such as holes and walls)
- [ ] Have different style of reward (more than just 1 / total trash)
- [ ] Multiple styles of action (Cardinal, Diagonal, etc)
- [ ] Better environment rendering (PyGame or OpenCV as well as text based)
- [ ] Ability to render the environment with a frame rate
- [ ] Ability to render the environment for a human to play
- [ ] Make the seed function properly seed the environment
- [ ] Ability to set custom boards
- [ ] Modifiers to player/trash value (instead of 1/2 for as_image=False and 122/255 for as_image=True)
- [ ] Register the env on inclusion
- [ ] Make a wrapper for registering the env that allows for kwargs

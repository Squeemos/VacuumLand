# VaccumLand
Gym environment for a trash finding robot

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
    - defualt = None
- seed
    - The seed for environment creation/reset
    - When None: Randomly seeds the environment when calling np.random.shuffle()
    - When Int: Custom value for seed, gets called during reset in np.random.seed(seed)
    - default = None

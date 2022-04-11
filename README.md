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

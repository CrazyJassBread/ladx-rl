# Running Zelda-LADX-v0

```bash
conda activate zelda
python -m pip install -e '.[test]'
make rom
make rom-test
make validate
```

`make rom` builds English US 1.0 in `ladx-disassembly/`. Its required MD5 is `07c211479386825042efb4ad31bb525f`.

```python
import gymnasium as gym
import zelda_env

env = gym.make("Zelda-LADX-v0", initial_state_path="save_states/azle.gbc.start.state", state_mode="reward", max_episode_steps=4500)
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(0)
env.close()
```

`info` contains `state`, `events`, `reward_terms`, `transition`, and `task`. Set `include_state_delta=True` when developing detectors. Inspect a state with:

```bash
python scripts/inspect_memory.py save_states/azle.gbc.start.state --mode debug
```

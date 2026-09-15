# Minimal ZeldaEnv API

`ZeldaEnv` implements the Gymnasium protocol with pixel observations and a
fixed 17-action `Discrete` space.

```python
from zelda_env import ZeldaEnv

env = ZeldaEnv(initial_state_path="save_states/azle.gbc.start.state")
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(1)

# Direct ZeldaEnv extension: override frame_skip for this transition.
observation, reward, terminated, truncated, info = env.step(1, num_frames=8)
```

The observation is always an RGB `uint8` array with shape `(144, 160, 3)`.
The default action order is `NOOP`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `A`, `B`,
`START`, `SELECT`, four directions with `A`, and four directions with `B`.

`info` has only the transition metadata needed by training and debugging:

```python
{
    "game_state": {...},
    "events": [...],
    "action": 1,
    "action_name": "UP",
    "num_frames": 4,
    "elapsed_steps": 1,
}
```

Additional direct helpers are available:

```python
frame = env.get_frame()
state = env.get_game_state()

checkpoint = env.save_state()
env.save_state("save_states/checkpoint.state")
env.load_state(checkpoint)                    # bytes or path

health = env.read_memory("wHealth")          # one byte -> int
items = env.read_memory("wInventoryItems", 12)  # multiple bytes -> bytes
```

`reset()` reloads `initial_state_path`, or the emulator's initial boot state if
no state was configured. A reset can override it with
`options={"state": bytes}` or `options={"state_path": path}`. It also accepts
an exact `options={"noop_frames": n}` advance for reproducible reset-time
variation; these frames do not count as environment steps.

`gym.make("Zelda-LADX-v0", ...)` supports the standard `step(action)` method.
Gymnasium wrappers do not forward the optional `num_frames` argument, so use
`ZeldaEnv` directly when that extension is needed.

To inspect a save state or find changing addresses:

```bash
python scripts/inspect_memory.py save_states/azle.gbc.start.state
python scripts/diff_memory.py before.state after.state
```

Run the visual random-policy test from Tail Cave room 1 with:

```bash
python examples/random_agent.py
```

The same window contains a scaled game frame on the left and a scrollable live
`info` panel on the right. Close the window or use `Ctrl-C` to stop. Options such
as `--steps`, `--frame-skip`, `--seed`, `--scale`, `--speed`,
`--include-menu-actions`, and `--headless` are available through `--help`.

For keyboard-controlled play instead of random actions:

```bash
python examples/human_play.py
```

Controls are arrow keys for movement, `Z` for A, `X` for B, `Enter` for START,
`Backspace` for SELECT, and `Esc` to quit. Direction + Z/X produces the matching
combined action from the fixed action space.

Task-specific rewards are added with `zelda_env.tasks.TaskEnv`, leaving this
base environment independent of individual rooms. See `docs/training.md` for
the Tail Cave PPO transfer suite.

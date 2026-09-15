# Zelda LADX RL Environment

This repository combines the LADX disassembly with a small Gymnasium
environment. `ladx-disassembly/` builds the ROM and `zelda_env/` controls it
through PyBoy.

The active project layout is intentionally small:

- `zelda_env/`: emulator adapter, Gymnasium environment, memory state and tasks;
- `configs/experiments/`: room/state splits and PPO settings;
- `training/` and `scripts/`: environment construction, training and evaluation;
- `examples/`: random and keyboard-controlled visual tests;
- `save_states/`: reproducible task starting points;
- `ladx-disassembly/`: upstream game source, build assets and generated ROM symbols.

## Quick start

```bash
conda activate zelda
make rom
python -m pytest -q
python scripts/validate_env.py
```

Use `make clean-cache` to remove Python/test caches without deleting ROMs,
save states, or experiment results.

The generated ROM must have MD5 `07c211479386825042efb4ad31bb525f`.

```python
from zelda_env import ZeldaEnv

env = ZeldaEnv(
    initial_state_path="save_states/azle.gbc.start.state",
    frame_skip=4,
    max_episode_steps=4_500,
)

observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(1)

game_state = info["game_state"]
print(game_state["room"], game_state["player"], game_state["monsters"])
env.close()
```

Observations are RGB pixels with shape `(144, 160, 3)`. Room, player,
inventory, event flags and active entity data are read from memory using
`ladx-disassembly/azle.sym` and returned in `info["game_state"]`.

See `docs/zelda_env_readme.md` for the complete API and
`ladx-disassembly/README.md` for disassembly build documentation.

To visually run a random policy from Tail Cave room 1:

```bash
python examples/random_agent.py
```

The window shows the live game frame on the left and the current `info` fields
in a scrollable panel on the right. Close the window or press `Ctrl-C` to stop.

For manual play with the same live info panel:

```bash
python examples/human_play.py
```

Use the arrow keys to move, `Z` for A, `X` for B, `Enter` for START,
`Backspace` for SELECT, and `Esc` to quit.

## Tail Cave PPO experiments

The A–E transfer suite trains from pixels while using RAM only for task rewards
and evaluation:

```bash
python -m pip install -e '.[train]'
python scripts/train.py --list
python scripts/train.py --experiment A --check
python scripts/train.py --experiment A
python scripts/evaluate.py artifacts/tail_cave/A/best_model.zip --experiment A
```

Experiment definitions are in
`configs/experiments/tail_cave_transfer.toml`. See `docs/training.md` for the
room splits, zero-shot protocol, and pretrained-versus-scratch fine-tuning
commands.

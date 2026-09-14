# Zelda RL Research Environment

This repository combines a preserved Link's Awakening DX disassembly with a
Gymnasium environment for memory-grounded reinforcement-learning research.

The project has three explicit layers:

- `ladx-disassembly/` builds the byte-exact English LADX ROM and symbol table.
- `zelda_env/` turns the ROM into a deterministic Gymnasium environment and
  maps emulator memory to semantic state and game events.
- `training/`, `configs/`, and `scripts/` contain experiment-facing tools.

## Quick start

```bash
conda activate zelda
make rom
make rom-test
python -m pytest -q
python scripts/validate_env.py
```

The ROM checksum must be `07c211479386825042efb4ad31bb525f`.
Generated ROMs are local artifacts and are not committed.

## Run the environment

```python
from zelda_env import ZeldaEnv

env = ZeldaEnv(
    initial_state_path="save_states/azle.gbc.start.state",
    state_mode="reward",
    max_episode_steps=4_500,
)
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(0)
env.close()
```

The policy observes pixels. Semantic memory state, state changes, detected game
events, and decomposed rewards are returned through `info`.

## Disassembly

The complete upstream-style disassembly remains independently buildable:

```bash
make -C ladx-disassembly build
make -C ladx-disassembly test
```

See [the preserved disassembly README](ladx-disassembly/README.md) for its
original build documentation.

## Documentation

- `docs/zelda_env_readme.md`: environment usage
- `docs/state_schema.md`: semantic state schema
- `docs/events.md`: state-delta and game-event model
- `docs/training.md`: task and training conventions

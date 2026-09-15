# Tail Cave PPO Transfer Experiments

The benchmark keeps the policy input pixel-only. RAM-derived `game_state` is
privileged information used exclusively for reward, termination and metrics.
Task logic lives under `zelda_env/tasks/`; room/state selection lives in
`configs/experiments/tail_cave_transfer.toml`.

## Experiment matrix

| ID | Train | Evaluate | Question |
|---|---|---|---|
| A | room `0x16`, fixed r2 state | same state | Can PPO solve the full kill-and-collect task? |
| B | room `0x16`, no-op frames 0–15 | no-op frames 16–30 | Does it generalize beyond one exact emulator frame? |
| C | two Hardhats in room `0x16` | one Hardhat in room `0x09` | Does the Hardhat-removal skill transfer zero-shot? |
| D | Hardhat and Keese kill-all rooms | held-out Spiked Beetle room | Does multi-room training generalize to a new clear-room mechanic? |
| E | room `0x09` | held-out room `0x09` state/jitter | Does C pretraining reduce fine-tuning samples versus scratch? |

B currently tests temporal variation because only one r2 save state exists.
Add new r2 states to the `states` arrays to extend it to position/layout
variation without changing Python code.

## Setup and validation

```bash
python -m pip install -e '.[train]'
python scripts/train.py --list
python scripts/train.py --experiment A --check
```

`--check` restores every referenced state and verifies its real room and target
entities before any training starts.

## Train and evaluate

```bash
python scripts/train.py --experiment A --seed 0
python scripts/evaluate.py artifacts/tail_cave/A/best_model.zip --experiment A
```

Use `--steps` for a smoke run or budget override:

```bash
python scripts/train.py -e C --steps 10000 --output artifacts/tail_cave/C_smoke
```

Each run writes `run.json`, a manifest snapshot, TensorBoard logs, periodic
checkpoints, `best_model.zip`, and `final_model.zip`. State SHA-256 values are
captured in `run.json`.

## Experiment E: fine-tune versus scratch

First train C, then run two E jobs with identical target-room settings:

```bash
python scripts/train.py -e C --output artifacts/tail_cave/C

python scripts/train.py -e E \
  --init-model artifacts/tail_cave/C/best_model.zip \
  --output artifacts/tail_cave/E_pretrained

python scripts/train.py -e E \
  --output artifacts/tail_cave/E_scratch

python scripts/evaluate.py artifacts/tail_cave/E_pretrained/best_model.zip \
  -e E --output artifacts/tail_cave/E_pretrained/result.json
python scripts/evaluate.py artifacts/tail_cave/E_scratch/best_model.zip \
  -e E --output artifacts/tail_cave/E_scratch/result.json
```

Run several seeds for meaningful comparisons. Primary metrics are success rate
and successful episode steps; TensorBoard curves show samples required to reach
a target success rate.

## Reward and success rules

The full r2 task tracks the two reset-time `0x20` entity slots. A target counts
as defeated when that slot disappears or changes type, which handles Hardhats
falling into pits without requiring their health byte to reach zero. The task
then waits for `wSmallKeysCount` to increase; merely seeing the `0x30` key entity
does not count as success.

The default shaping is intentionally small and room-independent:

- `-0.001` per step;
- `+1.0` once per removed target and `+0.5` when all targets are gone;
- `+0.2` when the key first appears and `+5.0` when collected;
- `-0.02` per health unit lost, `-2.0` on death;
- `-1.0` and termination when leaving the task room.

START and SELECT are excluded from training, while observations remain four
stacked RGB frames after the Stable-Baselines3 vector wrappers.

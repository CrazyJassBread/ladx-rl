# Tail Cave PPO Transfer Experiments

The benchmark keeps the policy input pixel-only. RAM-derived `game_state` is
privileged information used exclusively for reward, termination and metrics.
Task logic lives under `zelda_env/tasks/`; room/state selection lives in
`configs/experiments/tail_cave_transfer.toml`; reward weights live in
`configs/tasks/*.toml`.

Python task modules are organized by reusable mechanic rather than room:
`entity_task.py` tracks reset-time combat targets, `exit_task.py` handles
post-clear navigation, `key_task.py` handles key drops, and `chest_task.py`
owns the shared chest/reward/dialog lifecycle. Room-specific enemies, reward
amounts, destinations, and save states remain declarative TOML settings. This
keeps task semantics identical when the same skill is evaluated in a new room.

## Task and variant matrix

Experiment names use `task/variant`. The task name groups the room and semantic
objective; the variant describes the training or evaluation condition.

| Task | Variant | Train | Evaluate | Question |
|---|---|---|---|---|
| `room16_key` | `fixed` | room `0x16`, fixed r2 state | same state | Can PPO solve the full kill-and-collect task? |
| `room16_key` | `reset_jitter` | no-op frames 0–15 | no-op frames 16–30 | Does the same task generalize beyond one exact emulator frame? |
| `hardhat_transfer` | `room16_to_room09` | two Hardhats in room `0x16` | one Hardhat in room `0x09` | Does the Hardhat-removal skill transfer zero-shot? |
| `kill_all_transfer` | `room16_room12_to_room03` | Hardhat and Keese rooms | held-out Spiked Beetle room | Does multi-room training generalize to a new clear-room mechanic? |
| `room12_keese_exit` | `reset_jitter` | room `0x12`, no-op frames 0–15 | held-out no-op frames 16–30 | Can PPO defeat four Keese and leave through the upper door? |
| `room0d_moldorm_rupees` | `reset_jitter` | room `0x0D`, no-op frames 0–15 | held-out no-op frames 16–30 | Can PPO defeat the Mini Moldorm and finish the 20-Rupee chest dialog? |
| `room09_hardhat` | `reset_jitter` | room `0x09` | held-out room `0x09` reset jitter | Does room `0x16` pretraining improve fine-tuning versus scratch? |
| `room15_compass` | `reset_jitter` | four Hiding Zols in room `0x15` | held-out reset jitter in room `0x15` | Can PPO clear the room, open the chest, and acquire the Compass? |
| `room13_press_switch` | `curriculum` | room `0x13`, stop after switch activation | held-out reset jitter | Can PPO navigate, remove the blocking Hardhat, and hold the switch? |
| `room13_switch_chest` | `curriculum_finetune` | full room `0x13` task initialized from switch curriculum | held-out reset jitter | Can the learned switch policy extend to the chest? |
| `room13_switch_chest` | `reset_jitter` | shaped scratch baseline | held-out reset jitter | How much does curriculum initialization improve learning? |

`room16_key/reset_jitter` currently tests temporal variation because only one
r2 save state exists. Add new r2 states to the `states` arrays to extend it to
position/layout variation without changing Python code.

## Setup and validation

```bash
python -m pip install -e '.[train]'
python scripts/train.py --list
python scripts/train.py --experiment room16_key/fixed --check
```

`--check` restores every referenced state and verifies its real room and target
entities before any training starts.

For `tail_cave.r5.state`, the capture starts in room `0x12`. Train the complete
clear-and-exit objective with:

```bash
python scripts/train.py -e room12_keese_exit/reset_jitter --check
python scripts/train.py -e room12_keese_exit/reset_jitter --device cuda --num-envs 8
```

The task tracks exactly four reset-time `ENTITY_KEESE` slots. Defeating them
opens the shutter doors through room event `0x21`; clearing the enemies alone
does not end the episode. Success requires the RAM room tuple to change from
`(1, 0, 0x12)` to `(1, 0, 0x0D)`, the room directly above it in the Tail Cave
layout. Entering any other room is a failure.

The `tail_cave.r6.state` capture starts in room `0x0D`. Its complete task is:

```bash
python scripts/train.py -e room0d_moldorm_rupees/reset_jitter --check
python scripts/train.py -e room0d_moldorm_rupees/reset_jitter --device cuda --num-envs 8
```

The task tracks the single reset-time `ENTITY_MINI_MOLDORM` (`0x29`). Room
event `0x61` reveals a chest after the enemy is defeated, and the room's chest
table identifies its contents as `CHEST_RUPEES_20`. Success requires the Rupee
counter to increase by at least 20 and the seen `ENTITY_CHEST_WITH_ITEM` to
disappear after the item dialog closes. Merely starting the chest interaction
does not terminate the episode.

## Train and evaluate

```bash
python scripts/train.py --experiment room16_key/fixed --seed 0
python scripts/evaluate.py artifacts/tail_cave/room16_key/fixed/best_model.zip \
  --experiment room16_key/fixed
```

Evaluation saves the first episode for each evaluation instance as an animated
GIF. With the command above, GIFs are written under
`artifacts/tail_cave/room16_key/fixed/evaluation_gifs/`. The JSON report includes each GIF
path. Record more episodes or choose another directory with:

```bash
python scripts/evaluate.py artifacts/tail_cave/room16_key/fixed/best_model.zip \
  -e room16_key/fixed --record-episodes 3 \
  --gif-dir artifacts/tail_cave/room16_key/fixed/gifs
```

Set `--record-episodes 0` to disable recording. Playback speed defaults to the
effective emulator speed (`60 / frame_skip` FPS); `--gif-fps` overrides it.

The default `--device auto` uses `cuda:0` whenever CUDA is available and falls
back to CPU otherwise. On a GPU server, use strict CUDA selection to catch a
driver or PyTorch installation problem instead of silently training on CPU:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
python scripts/train.py --experiment room16_key/fixed --device cuda
```

Select another visible GPU with `--device cuda:1`. The resolved device, GPU
name, PyTorch version, and CUDA version are written to `run.json` for each run.

On a multi-core server, run independent PyBoy instances in subprocesses:

```bash
python scripts/train.py --experiment room16_key/fixed --device cuda --num-envs 8
```

`--num-envs` must be a multiple of the number of training instances. The
trainer reduces `n_steps` per worker so the total transitions per PPO update
stay equal to the single-worker baseline. For example,
`room16_key/fixed` changes from `1 x 512` to `8 x 64`, retaining 512
transitions per update. Environment
workers are assigned evenly across the configured training instances. The
actual worker count, per-worker steps, and rollout size are saved in
`run.json`.

Use `--steps` for a smoke run or budget override:

```bash
python scripts/train.py -e hardhat_transfer/room16_to_room09 --steps 10000 \
  --output artifacts/tail_cave/hardhat_transfer/smoke
```

Train the room `0x15` Compass task with parallel PyBoy workers:

```bash
python scripts/train.py -e room15_compass/reset_jitter --device cuda --num-envs 8
python scripts/evaluate.py \
  artifacts/tail_cave/room15_compass/reset_jitter/best_model.zip \
  -e room15_compass/reset_jitter
```

The `room15_compass/reset_jitter` task tracks exactly the four reset-time
`HIDING_ZOL` entities
(`0x9B`); the two `FIREBALL_SHOOTER` entities are room hazards and are not
targets. Clearing the enemies alone is not success. The agent must interact
with the central chest and cause the directly observed `wHasDungeonCompass`
flag to change. Success is delayed until the chest-item entity disappears, so
the pickup animation/dialog must finish rather than merely starting the chest
interaction. Leaving room `0x15` before completion is a failure.

Train the r4 room `0x13` task as a two-stage curriculum. Stage 1 learns the
safe route, Hardhat removal, and switch hold:

```bash
python scripts/train.py -e room13_press_switch/curriculum \
  --device cuda --num-envs 8
```

Stage 2 loads the best switch policy and fine-tunes the complete task:

```bash
python scripts/train.py -e room13_switch_chest/curriculum_finetune \
  --init-model artifacts/tail_cave/room13_press_switch/curriculum/best_model.zip \
  --device cuda --num-envs 8
python scripts/evaluate.py \
  artifacts/tail_cave/room13_switch_chest/curriculum_finetune/best_model.zip \
  -e room13_switch_chest/curriculum_finetune
```

The room event byte is `0x63`: `TRIGGER_STEP_ON_BUTTON |
EFFECT_REVEAL_CHEST`. Disassembly at bank `02:7810-781D` shows that
`wC1CA` must reach 24 consecutive frames before `wSwitchButtonPressed` becomes
`0x60`; the task therefore rewards signed hold progress instead of treating
the switch as a one-frame contact. `wRoomEventEffectExecuted` identifies the
chest reveal, and the task succeeds only after the chest entity appears and
`wSmallKeysCount` increases.

The shaped route was validated against the real r4 state. It goes around the
left side of the U-shaped pit to the upper corridor, pauses route shaping until
the reset-time `ENTITY_HARDHAT_BEETLE` slot is removed, then continues to the
button. The intended hazard-control action can push the Hardhat into a side
pit. The two Gel entities remain dynamic interference but are not mandatory
targets. After activation, the chest route returns to the upper corridor and
approaches the chest from below via the right side; a direct approach from
above is blocked.

Pit contact is detected from `wLinkGroundStatus`/`wPitSlippingCounter`;
entering `LINK_MOTION_FALLING_DOWN` is an immediate failure. Route coordinates,
weights, and the auxiliary success stage are TOML settings. They supervise
training through privileged RAM but are never included in the pixel policy
observation. Full episodes are capped at 1,200 steps; the switch curriculum is
capped at 800. Both use disjoint reset-jitter ranges.

Both curriculum stages use the same reduced 11-action space so their PPO
policy heads are checkpoint-compatible. Directional sword actions remain
available for the Hardhat and Gels, while redundant direction-plus-shield
actions are omitted; stationary `B` still exposes the shield.

Each run writes `run.json`, a manifest snapshot, TensorBoard logs, periodic
checkpoints, `best_model.zip`, and `final_model.zip`. State SHA-256 values are
captured in `run.json`.

## Room 0x09 Hardhat: fine-tune versus scratch

First train the source-room transfer variant, then run two room `0x09` jobs
with identical target-room settings:

```bash
python scripts/train.py -e hardhat_transfer/room16_to_room09

python scripts/train.py -e room09_hardhat/reset_jitter \
  --init-model artifacts/tail_cave/hardhat_transfer/room16_to_room09/best_model.zip \
  --output artifacts/tail_cave/room09_hardhat/pretrained

python scripts/train.py -e room09_hardhat/reset_jitter \
  --output artifacts/tail_cave/room09_hardhat/scratch

python scripts/evaluate.py artifacts/tail_cave/room09_hardhat/pretrained/best_model.zip \
  -e room09_hardhat/reset_jitter \
  --output artifacts/tail_cave/room09_hardhat/pretrained/result.json
python scripts/evaluate.py artifacts/tail_cave/room09_hardhat/scratch/best_model.zip \
  -e room09_hardhat/reset_jitter \
  --output artifacts/tail_cave/room09_hardhat/scratch/result.json
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

Each instance references a task TOML file. The current shaping is intentionally
small and room-independent:

- `-0.001` per step;
- `+1.0` once per removed target and `+0.5` when all targets are gone;
- `+0.2` when the key first appears and `+5.0` when collected;
- `-0.02` per health unit lost, `-2.0` on death;
- `-1.0` and termination when leaving the task room.

These values are no longer hard-coded in the task classes. See
`docs/task_rewards.md` for the signal catalog, TOML schema, and per-step
`reward_terms` diagnostics. Success and failure conditions remain independent
of reward weights.

START and SELECT are excluded from training, while observations remain four
stacked RGB frames after the Stable-Baselines3 vector wrappers.

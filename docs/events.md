# Events and rewards

Events are ordinary comparisons between the previous and current
`game_state`. `zelda_env/events.py` currently detects:

- room changes and first visits;
- player damage, healing and death;
- inventory, small-key, dungeon-key and instrument acquisition;
- event-flag changes;
- entity spawn, removal and damage;
- monster damage and defeat within the same room.

Every event is a JSON-safe dictionary:

```python
{"type": "player_damaged", "frame": 42, "data": {"amount": 8}}
```

`EVENT_REWARDS` is the small default event-to-reward table used by a bare
`ZeldaEnv`. Direct users can pass a replacement function:

```python
def reward(previous_state, current_state, events):
    return sum(1.0 for event in events if event["type"] == "monster_defeated")

env = ZeldaEnv(reward_fn=reward)
```

PPO experiments use `TaskEnv`, which intentionally replaces that base reward
with TOML-configured task rewards. See `docs/task_rewards.md`; do not edit
`EVENT_REWARDS` when tuning a training task.

Memory-address discovery does not run during training. The snapshot/diff tools
under `zelda_env/utils/` help locate candidate addresses, which should be
verified and then added to `zelda_env/memory.py`.

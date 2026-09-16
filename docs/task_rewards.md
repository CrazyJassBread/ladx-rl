# Task reward configuration

Training-task rewards are selected and weighted in `configs/tasks/*.toml`.
The Python runtime emits unweighted facts, while each task config decides which
facts contribute to its reward.

```toml
kind = "defeat_and_collect_item"
target_types = [155]
item_field = "dungeon_compass"
chest_type = 7

[reward]
step = -0.001
damage_taken = -0.02
player_died = -2.0
premature_room_exit = -1.0
target_defeated = 1.0
all_targets_cleared = 0.5
chest_revealed = 0.2
item_collected = 5.0
```

The experiment manifest references this file with `task_config`:

```toml
[instances.room15_compass_train]
states = ["save_states/azle.gbc.tail_cave.r3.state"]
expected_room = [1, 0, 21]
noop_frames = [0, 15]
task_config = "configs/tasks/room15_compass.toml"
```

## Signal vocabulary

| Signal | Value |
|---|---|
| `step` | `1` on every environment step |
| `damage_taken` | Health units lost during the transition |
| `player_died` | `1` when current health reaches zero |
| `premature_room_exit` | `1` when leaving the task room before success |
| `target_defeated` | Number of newly removed task-target slots |
| `all_targets_cleared` | `1` once, when the last target is removed |
| `destination_reached` | `1` when a task's required destination room is entered |
| `key_drop_seen` | `1` once, when the configured key drop first appears |
| `key_collected` | `1` when the small-key count first increases |
| `switch_pressed` | `1` once, when the dungeon floor-switch flag becomes non-zero |
| `switch_hold_progress` | Signed change in consecutive frames spent on the floor button |
| `route_progress` | Signed Manhattan-distance improvement toward the active TOML waypoint |
| `waypoint_reached` | `1` when the active route waypoint is reached |
| `blocking_hazard_removed` | Number of configured reset-time blocking hazards newly removed |
| `chest_revealed` | `1` once, when the room-event effect is executed after pressing the switch |
| `chest_opened` | `1` when a switch-chest task observes its small-key count increase |
| `item_collected` | `1` when the configured inventory flag first increases |
| `rupees_collected` | `1` once the configured Rupee increase has been reached |
| `dialog_completed` | `1` after a seen chest-item entity disappears when its dialog closes |
| `pit_contact` | `1` when Link first starts slipping over a pit |
| `fell_in_pit` | `1` when Link enters the falling-down motion state |

`zelda_env/reward_signals.py` owns this vocabulary and extracts the
task-independent signals. Task classes add goal-aware signals such as
`target_defeated` because they know which reset-time entity slots are targets.
Unknown TOML signal names are rejected during manifest loading.

The task wrapper returns the composed scalar reward and also exposes:

```python
info["task"]["reward_signals"]  # emitted unweighted values
info["task"]["reward_terms"]    # selected value * TOML weight
```

Success, failure, phases, and termination remain task logic rather than reward
configuration. Changing `item_collected = 5.0` therefore cannot change the
definition of task success.

Waypoints are privileged reward supervision, not observations. For example,
the room `0x13` config separates the safe route, the blocking-Hardhat stage,
and the chest route:

```toml
blocking_hazard_types = [32]
switch_waypoints = [[82, 107], [36, 107], [36, 27], [84, 27]]
post_hazard_waypoints = [[84, 27], [84, 59]]
chest_waypoints = [[84, 27], [119, 27], [119, 66], [139, 66], [139, 58]]
```

`route_progress` is signed, so moving away from a waypoint cancels prior
approach reward rather than allowing oscillation farming. The route pauses at
the upper corridor until the configured Hardhat slot disappears.

Each training run records task-config SHA-256 values in `run.json` and copies
the effective TOML files into the run's `task_configs/` directory.

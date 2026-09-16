# Game state

Pixel observations and privileged memory state are deliberately separate. The
agent receives a `(144, 160, 3)` RGB image; memory state is returned through
`info["game_state"]` for rewards and debugging.

```python
{
    "frame": 42,
    "room": {
        "id": 0x92,
        "map_id": 0,
        "indoor_room": 0,
        "is_indoor": 0,
        "is_side_scrolling": 0,
    },
    "player": {
        "x": 110,
        "y": 72,
        "z": 0,
        "direction": 2,
        "health": 24,
        "max_hearts": 3,
        "motion_state": 0,
        "ground_status": 0,
        "pit_slipping_counter": 0,
    },
    "inventory": {
        "items": [...],
        "tail_key": 0,
        "angler_key": 0,
        "face_key": 0,
        "bird_key": 0,
        "instruments": [...],
        ...
    },
    "progress": {...},
    "event_flags": {
        "switch_button_pressed": 0,
        "switch_button_hold_frames": 0,
        "room_event_executed": 0,
        ...
    },
    "entities": [...],
    "monsters": [...],
}
```

`entities` contains every active runtime entity slot. Each entry includes
`slot`, `status`, `type`, `x`, `y`, `z`, `health`, `direction`, and `room_id`.
`monsters` is a best-effort subset containing active entities with non-zero
health; `entities` is authoritative because some NPCs and special objects also
use that byte.

All mappings live in `zelda_env/memory.py` as simple semantic-name-to-symbol
dictionaries. Raw addresses are never duplicated in reward or event code.
`motion_state == 6` is the disassembly's `LINK_MOTION_FALLING_DOWN`, while
`ground_status == 7` is `GROUND_STATUS_PIT`. These privileged fields support
hazard metrics and termination but are not part of the policy observation.
`switch_button_hold_frames` maps the otherwise unlabeled `wC1CA`; its meaning
is statically confirmed by the floor-button code that increments it to 24
before writing `0x60` to `wSwitchButtonPressed`.

The static reference sheets under `docs/references/ladx/` are retained for
manual room and entity identification; runtime task logic must still use RAM
symbols and entity IDs rather than image coordinates from those sheets.

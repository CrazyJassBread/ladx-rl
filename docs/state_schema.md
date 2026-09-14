# Semantic State Schema v3

The policy observation is a `144 × 160 × 3` RGB frame. Memory-derived state is returned separately as `info["state"]`, so privileged RAM data does not silently become policy input.

Stable top-level keys are `meta`, `map`, `sprites`, `progress`, `effects`, and `flags`. `sprites.player.inventory` owns inventory data. Entity slots are exposed as `sprites.slots.slot_00` through `slot_0F`, with convenience views in `sprites.active` and `sprites.by_category`.

| Mode | Content | Intended use |
|---|---|---|
| `minimal` | player, location, progress | high-throughput experiments |
| `reward` | minimal plus entity slots | default rewards/events |
| `debug` | reward plus decoded room objects | memory-map development |
| `full` | debug plus raw entity tables | offline inspection only |

Legacy `world`, `player`, `inventory`, `entities`, and `room` aliases are only included with `include_legacy_aliases=True`.

Key mappings include `map.location.room ← hMapRoom`, `sprites.player.health.current ← wHealth`, `sprites.player.x/y ← hLinkPositionX/Y`, `sprites.player.inventory.items ← wInventoryItems`, and `sprites.slots.slot_XX.* ← wEntities*Table[slot]`.

Addresses come from `ladx-disassembly/azle.sym`; readable entity names come from `ladx-disassembly/src/constants/entities.asm`. Reward code consumes semantic paths/events and must not embed addresses.

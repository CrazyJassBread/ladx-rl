# State Deltas and Game Events

Each action advances PyBoy by `frame_skip` frames. The environment extracts semantic state, computes a leaf-level `StateDelta`, then passes both states to `LadxEventDetector`. Events contain `type`, `frame`, `data`, `source_paths`, and `confidence`.

Mapped events cover rooms, damage/healing/death, rupees and keys, heart pieces, instruments, inventory slots, entity spawn/damage/despawn/defeat, and room status. Entity identity combines slot, type, load order, and a detector-side generation. Despawn/defeat confidence is lower because RAM cannot always distinguish a kill from scripted removal.

`EventReward` and `TaskSpec` consume these names. Address-specific interpretation stays under `zelda_env/games/ladx/`. When adding an event, add a synthetic delta test and validate it from a named save state before assigning reward.

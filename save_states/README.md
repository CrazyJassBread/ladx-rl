# Save states

These PyBoy states are reproducible starting points for debugging and RL tasks.
The `rN` part of an older filename is a capture label, not necessarily the RAM
value of `hMapRoom`; use `python scripts/inspect_memory.py <state>` or the
experiment `--check` command to verify the actual room.

Rules for adding states:

- keep only byte-distinct states with a clear training or debugging purpose;
- reference training states from `configs/experiments/`;
- record important state hashes in `state_manifests/ladx_states.json`;
- never edit a state in place after it is used in a recorded experiment;
- let each training run capture the exact state hashes in its `run.json`.

`python scripts/verify_states.py` verifies the manifest and rejects byte-identical
state duplicates.

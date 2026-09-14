"""Compare LADX WRAM snapshots from two PyBoy save states."""

import argparse
from pathlib import Path

from zelda_env import ZeldaEnv
from zelda_env.memory.snapshot import MemorySnapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    env = ZeldaEnv(initial_state_path=args.before, state_mode="minimal")
    try:
        env.reset()
        before = MemorySnapshot.capture(env.backend)
        env.backend.load_state(Path(args.after).read_bytes())
        after = MemorySnapshot.capture(env.backend)
        changes = before.changed_addresses(after)
        for address, old, new in changes[: args.limit]:
            print(f"{address:04X}: {old:02X} -> {new:02X}")
        print(f"changed addresses: {len(changes)}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

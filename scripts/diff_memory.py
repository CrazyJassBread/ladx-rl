"""Compare LADX WRAM snapshots from two PyBoy save states."""

import argparse
from zelda_env import ZeldaEnv
from zelda_env.utils.memory_scanner import MemorySnapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    env = ZeldaEnv(initial_state_path=args.before)
    try:
        env.reset()
        before = MemorySnapshot.capture(env.emulator)
        env.load_state(args.after)
        after = MemorySnapshot.capture(env.emulator)
        changes = before.diff(after, env.symbols)
        for change in changes[: args.limit]:
            names = ", ".join(change["symbols"])
            suffix = f" ({names})" if names else ""
            print(f"{change['address']:04X}: {change['before']:02X} -> {change['after']:02X}{suffix}")
        print(f"changed addresses: {len(changes)}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

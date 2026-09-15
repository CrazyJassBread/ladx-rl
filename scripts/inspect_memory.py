"""Print a semantic state snapshot from a named emulator state."""

import argparse
import json

from zelda_env import ZeldaEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("state")
    args = parser.parse_args()
    env = ZeldaEnv(initial_state_path=args.state)
    try:
        _, info = env.reset()
        print(json.dumps(info["game_state"], indent=2, sort_keys=True))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

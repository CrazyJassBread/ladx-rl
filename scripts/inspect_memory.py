"""Print a semantic state snapshot from a named emulator state."""

import argparse
import json

from zelda_env import ZeldaEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("state")
    parser.add_argument("--mode", choices=("minimal", "reward", "debug", "full"), default="debug")
    args = parser.parse_args()
    env = ZeldaEnv(initial_state_path=args.state, state_mode=args.mode)
    try:
        _, info = env.reset()
        print(json.dumps(info["state"], indent=2, sort_keys=True))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

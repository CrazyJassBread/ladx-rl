"""Run a real-ROM smoke test and Gymnasium API validation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from gymnasium.utils.env_checker import check_env

from zelda_env import ZeldaEnv
from zelda_env.config import LadxPaths

EXPECTED_MD5 = "07c211479386825042efb4ad31bb525f"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="save_states/azle.gbc.start.state")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    paths = LadxPaths.default()
    digest = hashlib.md5(paths.rom_path.read_bytes()).hexdigest()
    if digest != EXPECTED_MD5:
        raise RuntimeError(f"ROM checksum mismatch: {digest}")
    kwargs = {"initial_state_path": Path(args.state), "max_episode_steps": args.steps + 1}
    env = ZeldaEnv(**kwargs)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
    env = ZeldaEnv(**kwargs)
    try:
        observation, info = env.reset(seed=0)
        assert observation.shape == (144, 160, 3)
        for _ in range(args.steps):
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                observation, info = env.reset(seed=0)
        assert info["state"]["meta"]["schema_version"] == 3
    finally:
        env.close()
    print(f"OK: checksum, Gym API, schema v3, {args.steps}-step PyBoy smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

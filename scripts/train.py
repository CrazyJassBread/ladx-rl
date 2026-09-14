"""Minimal PPO baseline; install with `pip install -e '.[train]'`."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--state", default="save_states/azle.gbc.start.state")
    parser.add_argument("--output", default="artifacts/ppo_ladx")
    args = parser.parse_args()
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[train]'") from exc
    from zelda_env import ZeldaEnv

    env = ZeldaEnv(initial_state_path=args.state, max_episode_steps=4500)
    try:
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="runs")
        model.learn(total_timesteps=args.steps)
        model.save(args.output)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

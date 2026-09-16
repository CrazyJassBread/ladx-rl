"""2D Zelda reinforcement-learning environment package."""

from zelda_env.env import ZeldaEnv
from zelda_env.memory import GameState

try:
    from gymnasium.envs.registration import register, registry

    if "Zelda-LADX-v0" not in registry:
        register(id="Zelda-LADX-v0", entry_point="zelda_env.env:ZeldaEnv")
except ImportError:  # optional dependency
    pass

__all__ = ["GameState", "ZeldaEnv"]

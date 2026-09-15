"""Central environment factory used by training and evaluation."""

from zelda_env import ZeldaEnv
from zelda_env.tasks.registry import make_task
from zelda_env.tasks.wrapper import TaskEnv

from training.experiments import InstanceConfig


def make_task_env(config: InstanceConfig) -> TaskEnv:
    """Build one pixel-only task instance from the experiment manifest."""

    env = ZeldaEnv(
        frame_skip=config.frame_skip,
        max_episode_steps=config.max_episode_steps,
    )
    task = make_task(config.task, task_id=config.name)
    return TaskEnv(
        env,
        task,
        config.state_paths,
        noop_frames=config.noop_frames,
        action_names=config.action_names,
        expected_room=config.expected_room,
    )

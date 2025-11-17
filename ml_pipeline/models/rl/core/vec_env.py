"""
Vectorized Environment Wrapper for Parallel Training

Runs multiple FundingArbitrageEnv instances in parallel using multiprocessing.
This significantly speeds up training by collecting rollouts from multiple
environments simultaneously.

Usage:
    envs = [lambda: create_environment(args, data_path) for _ in range(4)]
    vec_env = ParallelEnv(envs)
    obs = vec_env.reset()  # Returns (n_envs, obs_dim)
    next_obs, rewards, dones, truncated, infos = vec_env.step(actions)  # actions: (n_envs,)
"""

import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Callable, Any, Optional
import cloudpickle


def worker(remote, parent_remote, env_fn):
    """
    Worker process for running a single environment.

    Args:
        remote: Worker's end of the pipe
        parent_remote: Parent's end of the pipe
        env_fn: Function that creates the environment
    """
    parent_remote.close()
    env = env_fn()

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == 'step':
                obs, reward, terminated, truncated, info = env.step(data)
                # Auto-reset if episode ended
                if terminated or truncated:
                    # Store final info before reset
                    final_obs = obs
                    final_info = info.copy()
                    final_info['terminal_observation'] = final_obs
                    final_info['episode_ended'] = True

                    obs, reset_info = env.reset()
                    info.update(reset_info)
                    info['final_info'] = final_info
                else:
                    info['episode_ended'] = False

                remote.send((obs, reward, terminated, truncated, info))

            elif cmd == 'reset':
                obs, info = env.reset()
                remote.send((obs, info))

            elif cmd == 'get_action_mask':
                if hasattr(env, '_get_action_mask'):
                    mask = env._get_action_mask()
                else:
                    mask = None
                remote.send(mask)

            elif cmd == 'close':
                remote.close()
                break

            else:
                raise NotImplementedError(f"Unknown command: {cmd}")

    except KeyboardInterrupt:
        print(f"Worker received KeyboardInterrupt")
    finally:
        env.close()


class ParallelEnv:
    """
    Vectorized environment that runs multiple environments in parallel using multiprocessing.

    Each environment runs in its own process, allowing for true parallelism.
    """

    def __init__(self, env_fns: List[Callable], start_method: Optional[str] = None):
        """
        Initialize parallel environments.

        Args:
            env_fns: List of functions that create environments
            start_method: Multiprocessing start method ('spawn', 'fork', or None for default)
        """
        self.n_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])

        # Create worker processes
        ctx = mp.get_context(start_method)
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            # Serialize env_fn with cloudpickle to handle complex environments
            args = (work_remote, remote, cloudpickle.dumps(env_fn))
            process = ctx.Process(
                target=_worker_wrapper,
                args=args,
                daemon=True
            )
            process.start()
            self.processes.append(process)
            work_remote.close()

    def reset(self) -> Tuple[np.ndarray, List[dict]]:
        """
        Reset all environments.

        Returns:
            obs: Array of observations (n_envs, obs_dim)
            infos: List of info dicts
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.array(obs), list(infos)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Step all environments with the given actions.

        Args:
            actions: Array of actions (n_envs,)

        Returns:
            obs: Array of observations (n_envs, obs_dim)
            rewards: Array of rewards (n_envs,)
            terminated: Array of terminated flags (n_envs,)
            truncated: Array of truncated flags (n_envs,)
            infos: List of info dicts
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        results = [remote.recv() for remote in self.remotes]
        obs, rewards, terminated, truncated, infos = zip(*results)

        return (
            np.array(obs),
            np.array(rewards),
            np.array(terminated),
            np.array(truncated),
            list(infos)
        )

    def get_action_masks(self) -> np.ndarray:
        """
        Get action masks from all environments.

        Returns:
            Array of action masks (n_envs, n_actions) or None if not supported
        """
        for remote in self.remotes:
            remote.send(('get_action_mask', None))

        masks = [remote.recv() for remote in self.remotes]

        if all(mask is None for mask in masks):
            return None

        return np.array(masks)

    def close(self):
        """Close all environments and terminate worker processes."""
        if self.closed:
            return

        if self.waiting:
            for remote in self.remotes:
                remote.recv()

        for remote in self.remotes:
            remote.send(('close', None))

        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

        self.closed = True

    def __len__(self):
        return self.n_envs

    def __del__(self):
        if not self.closed:
            self.close()


def _worker_wrapper(work_remote, remote, env_fn_pickled):
    """
    Wrapper for worker that deserializes the environment function.

    This is necessary because cloudpickle doesn't work well with multiprocessing.Process
    when passing complex objects directly.
    """
    env_fn = cloudpickle.loads(env_fn_pickled)
    worker(work_remote, remote, env_fn)

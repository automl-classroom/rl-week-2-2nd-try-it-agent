from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType
from gymnasium.spaces import Discrete


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """Initializes the observation and action space for the environment."""
        self.observation_space = Discrete(2)
        self.action_space = Discrete(2)
        self.current_state = 0

    def reset(self, *, seed=None, options=None):
        self.current_state = 0
        return self.current_state, {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise RuntimeError(f"Action not valid: {action}")
        next_obs = int(
            np.argmax(self.get_transition_matrix()[self.current_state, action])
        )
        reward = self.get_reward_per_action()[self.current_state, action]
        terminated = False
        truncated = False
        self.current_state = next_obs
        return next_obs, reward, terminated, truncated, {}

    def get_reward_per_action(self):
        # reward = action
        reward_matrix = np.zeros((2, 2))
        reward_matrix[:, 0] = 0
        reward_matrix[:, 1] = 1
        return reward_matrix

    def get_transition_matrix(self):
        # state, action, state
        transition_matrix = np.zeros((2, 2, 2))
        transition_matrix[0, 0, 0] = 1
        transition_matrix[0, 1, 1] = 1
        transition_matrix[1, 0, 0] = 1
        transition_matrix[1, 1, 1] = 1
        return transition_matrix


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        super().__init__(env)
        self.current_state = 0
        assert 0.0 <= noise <= 1.0, "noise must be in [0,1]"
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        true_obs, info = self.env.reset(seed=seed, options=options)
        if self.rng.random() < self.noise:
            obs = 1 - true_obs
        else:
            obs = true_obs
        return obs, info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        true_obs, reward, terminated, truncated, info = self.env.step(action)
        if self.rng.random() < self.noise:
            obs = 1 - true_obs
        else:
            obs = true_obs
        return obs, float(reward), terminated, truncated, info

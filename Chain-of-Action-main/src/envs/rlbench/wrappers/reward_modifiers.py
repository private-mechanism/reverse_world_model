# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shape Rewards."""
import gymnasium as gym


class ShapeRewards(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Shape Rewards."""

    def __init__(self, env: gym.Env, reward_shaping_fn: callable):
        """General function to shape the rewards.

        Args:
            env: The environment to apply the wrapper
            reward_shaping_fn: The reward shaping function.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, reward_shaping_fn=reward_shaping_fn
        )
        gym.Wrapper.__init__(self, env)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.fn = reward_shaping_fn

    def step(self, action):
        """Steps through the environment, incrementing the time step.

        Args:
            action: The action to take

        Returns:
            The environment's step using the action.
        """
        observations, reward, *rest = self.env.step(action)
        return observations, self.fn(reward), *rest


class ScaleReward(ShapeRewards):
    """Scale Rewars."""

    def __init__(self, env: gym.Env, scale: float):
        """Scale the rewards.

        Args:
            env: The environment to apply the wrapper
            scale: The scale value
        """
        super().__init__(env, lambda r: r * scale)


class ClipReward(ShapeRewards):
    """Clip Rewards."""

    def __init__(self, env: gym.Env, lower_bound: float, upper_bound: float):
        """Clip the rewards.

        Args:
            env: The environment to apply the wrapper
            lower_bound: The lower bound
            upper_bound: The upper bound
        """
        super().__init__(env, lambda r: max(min(r, upper_bound), lower_bound))

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

"""Append Demo info."""
import gymnasium as gym
import numpy as np


class AppendDemoInfo(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Append a demo flag to the info dict."""

    def __init__(self, env: gym.Env):
        """Init.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def _modify_info(self, info):
        if "demo" not in info:
            if self.is_vector_env:
                info["demo"] = np.zeros((self.num_envs,))
            else:
                info["demo"] = 0
        return info

    def reset(self, *args, **kwargs):
        """See base."""
        obs, info = self.env.reset(*args, **kwargs)
        return obs, self._modify_info(info)

    def reset_to_demo(self, demo, *args, **kwargs):
        """See base."""
        obs, info = self.env.reset_to_demo(demo, *args, **kwargs)
        return obs, self._modify_info(info)
    
    def step(self, action):
        """See base."""
        *rest, info = self.env.step(action)
        return *rest, self._modify_info(info)

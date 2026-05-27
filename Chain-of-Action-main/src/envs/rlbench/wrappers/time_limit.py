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

from gymnasium.wrappers import TimeLimit


class TimeLimitX(TimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env, max_episode_steps)

    def reset_to_demo(self, demo, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.

        Args:
            demo: The demo to reset the environment with
            **kwargs: The kwargs to reset the environment with

        Returns:
            The reset environment
        """
        self._elapsed_steps = 0
        return self.env.reset_to_demo(demo, **kwargs)

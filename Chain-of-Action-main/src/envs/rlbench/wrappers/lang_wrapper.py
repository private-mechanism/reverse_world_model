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

"""Append language info and set randomized variation number."""
import gymnasium as gym
import numpy as np
import clip


class LangWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Randomize the variation number and language."""

    def __init__(self, env: gym.Env):
        """Init.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.tokenizer = clip.tokenize
        self.desc = None

    def reset(self, *args, **kwargs):
        """See base."""
        _env = self.env.unwrapped
        if hasattr(_env, "_task"):
            _env._task.sample_variation()
        obs, info = self.env.reset(*args, **kwargs)
        desc = info.pop("desc")
        desc = desc[np.random.randint(len(desc))]
        self.desc = self.tokenizer(desc).numpy()[0]

        return obs, {**info, "desc": self.desc}

    def reset_to_demo(self, demo, *args, **kwargs):
        """See base."""
        _env = self.env.unwrapped
        if hasattr(_env, "_task"):
            _env._task.sample_variation()
        obs, info = self.env.reset_to_demo(demo, *args, **kwargs)
        desc = info.pop("desc")
        desc = desc[np.random.randint(len(desc))]
        self.desc = self.tokenizer(desc).numpy()[0]

        return obs, {**info, "desc": self.desc}
    
    def step(self, action):
        """See base."""
        obs, *rest, info = self.env.step(action)
        return obs, *rest, {**info, "desc": self.desc}

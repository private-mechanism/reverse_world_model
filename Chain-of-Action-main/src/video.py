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

"""
Video recording utilities for RLBench evaluation.
"""
from pathlib import Path
from typing import Optional, Union
import imageio
import numpy as np
import gymnasium as gym


def _render_single_env_if_vector(env) -> Optional[np.ndarray]:
    """
    Render a single environment, handling both vector and single environments.
    
    Args:
        env: Environment to render
        
    Returns:
        Rendered frame as numpy array, or None if rendering fails
    """
    if getattr(env, "is_vector_env", False):
        if getattr(env, "parent_pipes", False):
            # Async vector env
            old_parent_pipes = env.parent_pipes
            env.parent_pipes = old_parent_pipes[:1]
            img = env.call("render")[0]
            env.parent_pipes = old_parent_pipes
        elif getattr(env, "envs", False):
            # Sync vector env
            old_envs = env.envs
            env.envs = old_envs[:1]
            img = env.call("render")[0]
            env.envs = old_envs
        else:
            raise ValueError("Unrecognized vector env.")
    else:
        img = env.render()
    return img


class VideoRecorder:
    """
    Video recorder for environment rendering during evaluation.
    
    This class handles recording frames from environment rendering and saving
    them as video files for evaluation analysis.
    """
    
    def __init__(self, save_dir: Optional[Union[str, Path]], render_size: int = 256, fps: int = 20):
        """
        Initialize video recorder.
        
        Args:
            save_dir: Directory to save videos. If None, recording is disabled.
            render_size: Size for rendering (not used in current implementation)
            fps: Frames per second for saved videos
        """
        self.save_dir = Path(save_dir) if save_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, env, enabled: bool = True) -> None:
        """
        Initialize recording for a new episode.
        
        Args:
            env: Environment to record
            enabled: Whether recording is enabled for this episode
        """
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env) -> None:
        """
        Record a single frame from the environment.
        
        Args:
            env: Environment to render and record
        """
        if self.enabled:
            frame = _render_single_env_if_vector(env)
            if frame is not None:
                self.frames.append(frame)

    def save(self, file_name: str) -> None:
        """
        Save recorded frames as a video file.
        
        Args:
            file_name: Name of the video file to save
        """
        if self.enabled and len(self.frames) > 0:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), np.array(self.frames), fps=self.fps) 

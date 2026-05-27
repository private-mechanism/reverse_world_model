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

import logging
import os
from datetime import datetime
from typing import Dict, Any
import numpy as np
import plotly.graph_objects as go
from omegaconf import OmegaConf

class Logger:
    """Log manager"""
    
    def __init__(self, cfg=None, log_dir=None):
        self.cfg = cfg
        # wandb related
        self._use_wandb = False
        self._wandb = None
        if cfg is not None and hasattr(cfg, 'wandb') and getattr(cfg.wandb, 'use', False):
            try:
                import wandb
                self._wandb = wandb
                self._use_wandb = True
                wandb.init(
                    project=cfg.wandb.project,
                    name=log_dir.name,
                    config=OmegaConf.to_container(cfg, resolve=False),
                    dir=cfg.wandb.save_dir,
                )
            except ImportError:
                self._use_wandb = False
                self._wandb = None
                self.warning('wandb is not installed, wandb logging is disabled.')
            except Exception as e:
                self._use_wandb = False
                self._wandb = None
                self.warning(f'wandb init failed: {e}')
    
        
        self.logger = logging.getLogger(__name__)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
        
    def log_metrics(self, metrics: Dict[str, Any], step: int = None, prefix: str = None):
        """Log training metrics and sync to wandb (if enabled)"""
        if step is not None:
            log_msg = f"Step {step}: "
        else:
            log_msg = "Metrics: "
            
        metric_strs = []
        wandb_log_dict = {}
        for key, value in metrics.items():
            # Add prefix to key
            if prefix is not None:
                wandb_key = f"{prefix}/{key}"
            else:
                wandb_key = key
                
            # torch tensor
            if hasattr(value, 'item'):
                value = value.item()
            
            # Special handling for video data
            if isinstance(value, dict) and ('video_success' in value or 'video_fail' in value):
                metric_strs.append(f"{key}=<video_data>")
                if self._use_wandb and self._wandb is not None:
                    try:
                        # Process success videos
                        if 'video_success' in value and len(value['video_success']) > 0:
                            video_success_list = []
                            for video in value['video_success']:
                                if video.ndim == 4:
                                    video_transposed = video.transpose(0, 3, 1, 2)
                                    fps = value.get('fps', 10)
                                    wandb_video = self._wandb.Video(
                                        video_transposed, step, fps=fps, format="mp4"
                                    )
                                    video_success_list.append(wandb_video)
                            if video_success_list:
                                wandb_log_dict[f"{wandb_key}_success"] = video_success_list
                        
                        # Process failure videos
                        if 'video_fail' in value and len(value['video_fail']) > 0:
                            video_fail_list = []
                            for video in value['video_fail']:
                                if video.ndim == 4:
                                    video_transposed = video.transpose(0, 3, 1, 2)
                                    fps = value.get('fps', 10)
                                    wandb_video = self._wandb.Video(
                                        video_transposed, step, fps=fps, format="mp4"
                                    )
                                    video_fail_list.append(wandb_video)
                            if video_fail_list:
                                wandb_log_dict[f"{wandb_key}_fail"] = video_fail_list
                    except Exception as e:
                        self.warning(f'wandb video log failed: {e}')
                continue
            
            # Only use .6f formatting for numeric types
            if isinstance(value, (int, float)):
                metric_strs.append(f"{key}={value:.6f}")
                wandb_log_dict[wandb_key] = value
            elif isinstance(value, str):
                metric_strs.append(f"{key}={value}")
                # Image/video files automatically uploaded to wandb
                if self._use_wandb and self._wandb is not None:
                    try:
                        if value.endswith('.png'):
                            wandb_log_dict[wandb_key] = self._wandb.Image(value)
                        elif value.endswith('.mp4'):
                            wandb_log_dict[wandb_key] = self._wandb.Video(value)
                        else:
                            wandb_log_dict[wandb_key] = value
                    except Exception as e:
                        self.warning(f'wandb media log failed: {e}')
                else:
                    wandb_log_dict[wandb_key] = value
            # plotly figures automatically uploaded to wandb
            elif isinstance(value, go.Figure):
                metric_strs.append(f"{key}=<plotly.Figure>")
                if self._use_wandb and self._wandb is not None:
                    try:
                        wandb_log_dict[wandb_key] = self._wandb.Plotly(value)
                    except Exception as e:
                        self.warning(f'wandb plotly log failed: {e}')
            # numpy arrays automatically uploaded as images/videos
            elif isinstance(value, np.ndarray):
                metric_strs.append(f"{key}=<np.ndarray shape={value.shape}>")
                if self._use_wandb and self._wandb is not None:
                    try:
                        if value.ndim == 4:
                            wandb_log_dict[wandb_key] = self._wandb.Image(value)
                        elif value.ndim == 5:
                            wandb_log_dict[wandb_key] = self._wandb.Video(value, fps=10, format="mp4")
                    except Exception as e:
                        self.warning(f'wandb numpy media log failed: {e}')
            else:
                metric_strs.append(f"{key}={str(value)[:50]}...")  # Truncate long strings
        log_msg += ", ".join(metric_strs)
        self.info(log_msg)
        # wandb log synchronization
        if self._use_wandb and self._wandb is not None and len(wandb_log_dict) > 0:
            try:
                if step is not None:
                    self._wandb.log(wandb_log_dict, step=step)
                else:
                    self._wandb.log(wandb_log_dict)
            except Exception as e:
                self.warning(f'wandb.log failed: {e}') 

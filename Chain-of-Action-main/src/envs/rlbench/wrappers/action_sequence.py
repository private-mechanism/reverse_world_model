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

"""Wrapper for allowing action sequences."""

from typing import Any, Dict
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


class ActionSequence(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Wrapper for allowing action sequences."""

    def __init__(self, env: gym.Env, sequence_length: int):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)
        self._sequence_length = sequence_length
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.is_demo_env = getattr(env, "is_demo_env", False)
        if self.is_vector_env:
            raise NotImplementedError(
                "It is not possible to use this wrapper with a VecEnv."
            )
        low, high = env.action_space.low, env.action_space.high
        self.action_space = Box(
            np.expand_dims(low, 0).repeat(sequence_length, 0),
            np.expand_dims(high, 0).repeat(sequence_length, 0),
            dtype=self.action_space.dtype,
        )

    def _step_sequence(self, action):
        total_reward = np.array(0.0)
        action_idx_reached = 0
        if self.is_demo_env:
            demo_actions = np.array(action)
        for i, sub_action in enumerate(action):
            if sub_action.sum() == 0:
                break
            observation, reward, termination, truncation, info = self.env.step(
                sub_action
            )
            # if i==0:
            #     observation_sequence = observation
            # else: 
            #     observation_sequence['front_rgb'] = np.concatenate([observation_sequence['front_rgb'], observation['front_rgb']], axis=0)
            # cv2.imwrite(f"debug/t_{i}.png",np.moveaxis(observation['front_rgb'][0],0,2))
            if self.is_demo_env:
                demo_actions[i] = info.pop("demo_action")
            total_reward += reward
            action_idx_reached += 1
            if termination or truncation:
                break
        assert action_idx_reached <= self._sequence_length
        info["action_sequence_mask"] = (
            np.arange(self._sequence_length) < action_idx_reached
        ).astype(int)
        if self.is_demo_env:
            info["demo_action"] = np.array(demo_actions)
        return observation, total_reward, termination, truncation, info

    def step(self, action):
        # if action.shape != self.action_space.shape:
        #     raise ValueError(
        #         f"Expected action to be of shape {self.action_space.shape}, "
        #         f"but got action of shape {action.shape}."
        #     )
        return self._step_sequence(action)


class TemporalEnsemble(ActionSequence):
    """Receding horizon control with temporal ensembling of ACT.

    This wrapper allows agent predict an action sequence of length N,
    but performs receding horizon control of only K <= N steps of actions.
    We also support temporal ensembling (from ALOHA https://arxiv.org/abs/2304.13705),
    which caches the previous actions and outputs a weighted average of them.
    """

    def __init__(
        self,
        env: gym.Env,
        sequence_length: int,
        time_limit: int,
        execution_length: int,
        temporal_ensemble: bool = True,
        gain: float = 0.01,
        escape_threshold: float = 0.005
    ):
        """Init.

        Args:
            env: The gym env to wrap.
            sequence_length: Action sequence length.
            time_limit: The time limit of the env for creating buffers.
            execution_length: The execution length of the receding horizion control.
            temporal_ensemble: Whether to use temporal ensembling. Defaults to True.
            gain: Temporal ensembling gain. Defaults to 0.01.
        """
        super().__init__(env, sequence_length)
        self._time_limit = time_limit
        self._execution_length = execution_length
        self._temporal_ensemble = temporal_ensemble
        self._gain = gain
        self._init_action_history()
        self.escape_threshold = escape_threshold

    def _init_action_history(self):
        """Initialize the action history buffer.

        We store the history actions within a buffer of shape [T, T + L, A],
        where T is the time limit, L is the sequence length, and A is the action size.

        For example, self._action_history[t, t:t + L] stores the predicted action
        sequence of size A and length L at time step t.
        """
        self._action_history = np.zeros(
            [
                self._time_limit,
                self._time_limit + self._sequence_length,
                self.action_space.shape[-1],
            ],
            dtype=self.action_space.dtype,
        )
        self._cur_step = 0

    def reset(
        self, *, seed: int = None, options: Dict[str, Any] = None
    ) -> tuple[Any, dict[str, Any]]:
        self._init_action_history()
        self.previous_action = None
        self.stuck_times = 0
        return super().reset(seed=seed, options=options)

    def reset_to_demo(
        self, demo, *, seed: int = None, options: Dict[str, Any] = None
    ) -> tuple[Any, dict[str, Any]]:
        self._init_action_history()
        self.previous_action = None
        self.stuck_times = 0
        return self.env.reset_to_demo(demo, seed=seed, options=options)
    
    def _step_sequence(self, action):
        total_reward = np.array(0.0)
        action_idx_reached = 0
        if self.is_demo_env:
            demo_actions = np.array(action)

        # assert self._execution_length ==1, "Only support execution_length=1 for now."
        for i, sub_action in enumerate(action):
            if self._temporal_ensemble and self._sequence_length > 1:
                self._action_history[
                    self._cur_step, self._cur_step : self._cur_step + self._sequence_length
                ] = action
                # Select all predicted actions for self._cur_step. This will cover the
                # actions from [cur_step - sequence_length + 1, cur_step)
                # Note that not all actions in this range will be valid as we might have
                # execution_length > 1, which skips some of the intermediate steps.
                cur_actions = self._action_history[:, self._cur_step]
                indices = np.all(cur_actions != 0, axis=1)
                cur_actions = cur_actions[indices]

                # earlier predicted actions will have smaller weights.
                exp_weights = np.exp(-self._gain * np.arange(len(cur_actions)))
                exp_weights = (exp_weights / exp_weights.sum())[:, None]
                sub_action = (cur_actions * exp_weights).sum(axis=0)
            
            # if self.previous_action is not None and np.linalg.norm((self.previous_action-sub_action)[:]) < self.escape_threshold:
                # if i < len(action) - 1:
                #     sub_action = action[i+1]
                # sub_action = action[-1]
                # self.stuck_times+=1
                # if self.stuck_times > 5:
                #     sub_action = action[i+1]
                #     self.stuck_times=0
                
            observation, reward, termination, truncation, info = self.env.step(
                sub_action
            )
            # if i==0:
            #     observation_sequence = observation
            # else: 
            #     observation_sequence['front_rgb'] = np.concatenate([observation_sequence['front_rgb'], observation['front_rgb']], axis=0)
            # cv2.imwrite(f"debug/t_{i}.png",np.moveaxis(observation['front_rgb'][0],0,2))

            self._cur_step += 1
            if self.is_demo_env:
                demo_actions[i] = info.pop("demo_action")
            total_reward += reward
            action_idx_reached += 1
            if termination or truncation:
                break

            if not self.is_demo_env:
                if action_idx_reached == self._execution_length:
                    break

            if sub_action.sum() == 0:
                break

        assert action_idx_reached <= self._sequence_length
        # TODO not sure this is correct in the case of receding horizon control
        #      Currently, for every action_sequence, all actions that are not applied
        #      will be masked out!!
        # info["action_sequence_mask"] = (
        #     np.arange(self._sequence_length) < action_idx_reached
        # ).astype(int)
        if self.is_demo_env:
            info["demo_action"] = np.array(demo_actions)
        self.previous_action = sub_action
        return (
            observation,
            total_reward,
            termination,
            truncation,
            info,
        )

class ReverseTemporalEnsemble(ActionSequence):
    """Receding horizon control with temporal ensembling of ACT.

    This wrapper allows agent predict an action sequence of length N,
    but performs receding horizon control of only K <= N steps of actions.
    We also support temporal ensembling (from ALOHA https://arxiv.org/abs/2304.13705),
    which caches the previous actions and outputs a weighted average of them.
    """

    def __init__(
        self,
        env: gym.Env,
        sequence_length: int,
        time_limit: int,
        execution_length: int,
        temporal_ensemble: bool = True,
        gain: float = 0.01,
        action_order: str='REVERSE'
    ):
        """Init.

        Args:
            env: The gym env to wrap.
            sequence_length: Action sequence length.
            time_limit: The time limit of the env for creating buffers.
            execution_length: The execution length of the receding horizion control.
            temporal_ensemble: Whether to use temporal ensembling. Defaults to True.
            gain: Temporal ensembling gain. Defaults to 0.01.
        """
        super().__init__(env, sequence_length)
        self._time_limit = time_limit
        self._execution_length = execution_length
        self._temporal_ensemble = temporal_ensemble
        self._gain = gain
        self._init_action_history()
        self.action_order = action_order

    def _init_action_history(self):
        """Initialize the action history buffer.

        We store the history actions within a buffer of shape [T, T + L, A],
        where T is the time limit, L is the sequence length, and A is the action size.

        For example, self._action_history[t, t:t + L] stores the predicted action
        sequence of size A and length L at time step t.
        """
        self._action_history = np.zeros(
            [
                self._time_limit,
                self._sequence_length,
                self.action_space.shape[-1],
            ],
            dtype=self.action_space.dtype,
        )
        self._cur_step = 0

    def reset(
        self, *, seed: int = None, options: Dict[str, Any] = None
    ) -> tuple[Any, dict[str, Any]]:
        self._init_action_history()
        self.previous_action = None
        self.previous_v = None
        return super().reset(seed=seed, options=options)

    def reset_to_demo(
        self, demo, *, seed: int = None, options: Dict[str, Any] = None
    ) -> tuple[Any, dict[str, Any]]:
        self._init_action_history()
        self.previous_action = None
        self.previous_v = None
        return self.env.reset_to_demo(demo, seed=seed, options=options)
    
    def _step_sequence(self, action):
        total_reward = np.array(0.0)
        action_idx_reached = 0
        if self.is_demo_env:
            demo_actions = np.array(action)



        # assert self._execution_l ength ==1, "Only support execution_length=1 for now."
        if  self.action_order == 'REVERSE' or self.action_order == 'HYBRID': 
            if not self._temporal_ensemble:
                if self.previous_v is not None:
                    current_v = (action[-1]-action[0])[:3]
                    # while iteration increase the action step, once the movement is enough, then it stops
                    while np.linalg.norm(self.previous_v) < 0.002 + np.linalg.norm(current_v) and len(action)>1:
                        action = action[1:]
                        current_v = (action[-1]-action[0])[:3]
            self._action_history[self._cur_step, -len(action):] = action

        #reset action history when reach nbp

            
        self._cur_step_inner = 0
        # assert self._execution_length ==1, "Only support execution_length=1 for now."
        for i, sub_action in enumerate(action):
            if  self.action_order == 'REVERSE' or self.action_order == 'HYBRID':     
                if self._temporal_ensemble and self._sequence_length > 1:
                    while True:
                        # Select all predicted actions for self._cur_step. This will cover the
                        # actions from [cur_step - sequence_length + 1, cur_step)
                        # Note that not all actions in this range will be valid as we might have
                        # execution_length > 1, which skips some of the intermediate steps.
                        cur_actions = self._action_history[:, -len(action) + self._cur_step_inner + i]
                        indices = np.all(cur_actions != 0, axis=1)
                        cur_actions = cur_actions[indices]

                        # earlier predicted actions will have smaller weights.
                        exp_weights = np.exp(-self._gain * np.arange(len(cur_actions)))
                        exp_weights = (exp_weights / exp_weights.sum())[:, None]
                        sub_action = (cur_actions * exp_weights).sum(axis=0)

                        if self.previous_v is None :
                            break
                        
                        current_v = (action[-1]-sub_action)[:3]
                        if np.linalg.norm(self.previous_v) > 0.002 + np.linalg.norm(current_v) and (self.previous_v * current_v).sum()>0:
                            break
                        elif -len(action) + self._cur_step_inner + i < -1:
                            self._cur_step_inner +=1
                            continue
                        else: 
                            break

            
            # print(sub_action)
            observation, reward, termination, truncation, info = self.env.step(
                sub_action
            )
            # img = np.moveaxis(observation['front_rgb'][0], 0, 2)
            # self.table.add_data(wandb.Image(img),self._cur_step, self._cur_step_inner,len(action),np.array2string(sub_action),np.array2string(action))
            # wandb.log({'sub_action':sub_action},step=self._cur_step_inner)
            
            # if i==0:
            #     observation_sequence = observation
            # else:
            #     observation_sequence['front_rgb'] = np.concatenate([observation_sequence['front_rgb'], observation['front_rgb']], axis=0)
            # cv2.imwrite(f"debug/t_{self._cur_step}.png",np.moveaxis(observation['front_rgb'][0],0,2))
            
        
            self._cur_step_inner +=1
            if self.is_demo_env:
                demo_actions[i] = info.pop("demo_action")
            total_reward += reward
            action_idx_reached += 1
            if termination or truncation:
                break

            if not self.is_demo_env:
                if action_idx_reached == self._execution_length:
                    break

            if sub_action.sum() == 0:
                break
        
        self._cur_step += 1

        if self.action_order == 'REVERSE' or self.action_order == 'HYBRID': 
            if self._temporal_ensemble:
                self.previous_v = (action[-1] - sub_action)[:3] # TODO: use action[-1] in history
                if -len(action) + self._cur_step_inner + i == 0:
                    self._init_action_history()
                    self.previous_v = None
            else:
                self.previous_v = (action[-1] - sub_action)[:3]
                if len(action) == 1:
                    self.previous_v = None

        assert action_idx_reached <= self._sequence_length
        # TODO not sure this is correct in the case of receding horizon control
        #      Currently, for every action_sequence, all actions that are not applied
        #      will be masked out!!
        # info["action_sequence_mask"] = (
        #     np.arange(self._sequence_length) < action_idx_reached
        # ).astype(int)
        if self.is_demo_env:
            info["demo_action"] = np.array(demo_actions)
        # self.previous_action = sub_action
        
        
        return (
            observation,
            total_reward,
            termination,
            truncation,
            info,
        )

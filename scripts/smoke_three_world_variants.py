#!/usr/bin/env python3
"""Lightweight CPU smoke checks for the three supported WVWAM variants."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from wam.models.mvwam.build_model import (
    _strip_to_world_policy_keys,
    load_wam_checkpoint_into_world_policy,
)
from wam.runners.runner_world_casual_128dim import (
    FMPRunner,
    resolve_world_generation_variant,
)
from wam.samples.noise_scheduler import NoiseTimestepSampler


class FakeVideoScheduler:
    def set_timesteps(self, count, device=None):
        self.timesteps = torch.linspace(900, 100, count, device=device)

    def step(self, prediction, _timestep, sample, return_dict=False):
        return (sample - prediction * 0.01,)


class FakeWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, *, hidden_states_video=None, hidden_states_action=None, **_kwargs):
        self.calls.append(
            (
                hidden_states_video is not None,
                hidden_states_action is not None,
            )
        )
        video = (
            torch.zeros_like(hidden_states_video[:, :16])
            if hidden_states_video is not None
            else None
        )
        action = (
            torch.zeros_like(hidden_states_action[:, :, :128])
            if hidden_states_action is not None
            else None
        )
        return video, action


def make_runner(variant: str) -> FMPRunner:
    runner = FMPRunner.__new__(FMPRunner)
    nn.Module.__init__(runner)
    runner.model = FakeWorld()
    values = {
        "dtype": torch.float32,
        "_video_expert_is_wan22": False,
        "_noise_video_channels": 16,
        "state_token_dim": 128,
        "pred_horizon": 8,
        "key_action_chunk_size": 2,
        "use_value": False,
        "prediction_type": "sample",
        "noise_scheduler_type": "flow",
        "num_inference_timesteps": 2,
        "world_generation_variant": variant,
        "goal_conditioned_wam": variant != "original",
        "predict_key_video": True,
        "predict_key_action": True,
        "goal_joint_key_diffusion": variant == "goal_reverse_joint",
        "goal_joint_reverse_diffusion": variant == "goal_reverse_joint",
        "use_predicted_key_condition": True,
        "detach_predicted_key_condition": True,
        "reverse_video_key_condition_type": "latent_inpainting",
        "reverse_action_key_condition_type": "prefix",
        "key_video_loss_weight": 1.0,
        "key_action_loss_weight": 1.0,
        "reverse_video_loss_weight": 1.0,
        "reverse_action_loss_weight": 1.0,
        "goal_conditioned_video_guidance_scale": 1.0,
        "goal_conditioned_return_forward_actions": True,
        "reverse_world_order": variant != "original",
        "last_goal_conditioned_metrics": {},
    }
    for name, value in values.items():
        setattr(runner, name, value)
    runner.noise_scheduler_action = NoiseTimestepSampler(num_train_timesteps=1000)
    runner.noise_scheduler_video = NoiseTimestepSampler(num_train_timesteps=1000)
    runner.noise_scheduler_sample_action = NoiseTimestepSampler(
        num_train_timesteps=1000
    )
    runner.noise_scheduler_sample_video = FakeVideoScheduler()
    return runner


def make_inputs():
    batch, frames, height, width = 2, 4, 2, 2
    return {
        "lang_tokens": torch.randn(batch, 8, 32),
        "lang_attn_mask": torch.ones(batch, 8, dtype=torch.bool),
        "img_tokens": torch.randn(batch, 6, 24),
        "state_tokens": torch.randn(batch, 1, 128),
        "action_gt": torch.randn(batch, 8, 128),
        "action_mask": torch.ones(batch, 1, 128),
        "ctrl_freqs": torch.ones(batch),
        "video_latents": torch.randn(batch, 16, frames, height, width),
        "condition_video_latents": torch.zeros(batch, 20, frames, height, width),
        "action_valid_mask": torch.ones(batch, 8),
    }


def check_training_paths():
    inputs = make_inputs()
    expected_calls = {
        "original": [(True, True)],
        "goal_reverse_independent": [
            (True, False),
            (False, True),
            (True, False),
            (False, True),
        ],
        "goal_reverse_joint": [(True, True), (True, True)],
    }
    for variant, expected in expected_calls.items():
        runner = make_runner(variant)
        loss_action, loss_video, loss_value = runner.compute_loss(**inputs)
        assert loss_action.ndim == 0 and loss_video.ndim == 0
        assert loss_value is None
        assert runner.model.calls == expected, (variant, runner.model.calls)


def check_goal_inference_paths():
    inputs = make_inputs()
    for variant, call_count in (
        ("goal_reverse_independent", 8),
        ("goal_reverse_joint", 4),
    ):
        runner = make_runner(variant)
        out = runner.conditional_sample_goal_conditioned_reverse_diffusion(
            lang_tokens=inputs["lang_tokens"],
            lang_attn_mask=inputs["lang_attn_mask"],
            img_tokens=inputs["img_tokens"],
            state_tokens=inputs["state_tokens"],
            action_mask=inputs["action_mask"],
            ctrl_freqs=inputs["ctrl_freqs"],
            video_latents=inputs["video_latents"],
            condition_video_latents=inputs["condition_video_latents"],
            video_only=False,
            action_only=False,
            seed=7,
            num_inference_timesteps=2,
        )
        assert len(runner.model.calls) == call_count
        assert torch.equal(out["pred_video"][:, :, :1], out["pred_key_video"])
        assert torch.equal(
            out["pred_reverse_trajectory"][:, :2], out["pred_key_action"]
        )
        assert torch.equal(
            out["pred_trajectory"],
            torch.flip(out["pred_reverse_trajectory"], dims=[1]),
        )
        assert out["trajectory_order"] == "forward"


def check_legacy_checkpoint_filter():
    legacy = {
        "module.model.video_expert.block.weight": torch.ones(1),
        "model.action_expert.block.weight": torch.ones(1),
        "stage4_video_ar_head.0.weight": torch.ones(1),
        "module.stage3_decoder.weight": torch.ones(1),
    }
    filtered = _strip_to_world_policy_keys(legacy)
    assert set(filtered) == {
        "video_expert.block.weight",
        "action_expert.block.weight",
    }

    class TinyWorld(nn.Module):
        def __init__(self):
            super().__init__()
            self.video_expert = nn.Linear(2, 2, bias=False)
            self.action_expert = nn.Linear(2, 2, bias=False)

    world = TinyWorld()
    video_weight = torch.full_like(world.video_expert.weight, 2.0)
    action_weight = torch.full_like(world.action_expert.weight, 3.0)
    checkpoint = {
        "module.model.video_expert.weight": video_weight,
        "module.model.action_expert.weight": action_weight,
        "module.stage4_video_ar_head.0.weight": torch.ones(1),
    }
    with tempfile.TemporaryDirectory() as tmp:
        torch.save(checkpoint, Path(tmp) / "pytorch_model.bin")
        load_wam_checkpoint_into_world_policy(world, tmp)
    assert torch.equal(world.video_expert.weight, video_weight)
    assert torch.equal(world.action_expert.weight, action_weight)


def check_legacy_config_mapping():
    assert resolve_world_generation_variant({}) == "original"
    assert (
        resolve_world_generation_variant({"goal_conditioned_wam": True})
        == "goal_reverse_independent"
    )
    assert (
        resolve_world_generation_variant(
            {
                "goal_conditioned_wam": True,
                "goal_joint_key_diffusion": True,
            }
        )
        == "goal_reverse_joint"
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    check_training_paths()
    check_goal_inference_paths()
    check_legacy_checkpoint_filter()
    check_legacy_config_mapping()
    print("three WVWAM variants smoke test: PASS")

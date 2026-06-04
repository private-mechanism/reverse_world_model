"""RoboTwin policy adapter for WVWAM online success-rate evaluation.

Copy or symlink this directory to ``<RoboTwin>/policy/WVWAM`` and set
``policy_name: WVWAM.deploy_policy`` in the RoboTwin eval config.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping


DEFAULT_WAM_REPO = "/mnt/world_foundational_model/fyzhao/reverse_world_model"


def _get_arg(args: Any, key: str, default=None):
    if isinstance(args, Mapping):
        return args.get(key, default)
    return getattr(args, key, default)


def _add_wam_repo_to_path(usr_args: Any) -> None:
    repo = (
        _get_arg(usr_args, "wam_repo", None)
        or _get_arg(usr_args, "wvwam_repo", None)
        or os.environ.get("WAM_REPO")
        or DEFAULT_WAM_REPO
    )
    repo_path = Path(str(repo)).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(
            f"WAM repo path does not exist: {repo_path}. "
            "Set WAM_REPO or pass wam_repo in RoboTwin usr_args."
        )
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def get_model(usr_args):
    """RoboTwin calls this once before evaluation episodes start."""
    _add_wam_repo_to_path(usr_args)
    from wam.deploy.robotwin_success_rate import load_policy_from_args

    return load_policy_from_args(usr_args)


def encode_obs(observation):
    """Optional RoboTwin hook; WVWAM preprocessing is done inside the policy."""
    return observation


def get_action(model, observation, instruction=None):
    """Optional RoboTwin hook for versions that split action selection."""
    return model.act(observation, instruction=instruction)


def update_obs(model, observation):
    """Optional RoboTwin hook for versions that maintain an observation window."""
    return None


def eval(TASK_ENV, model, observation, instruction=None):
    """RoboTwin calls this inside its online rollout loop."""
    from wam.deploy.robotwin_success_rate import take_action_with_robotwin

    if instruction is not None and hasattr(model, "deploy_config"):
        model.deploy_config.instruction = instruction
    return take_action_with_robotwin(TASK_ENV, model, encode_obs(observation))


def reset_model(model=None):
    """RoboTwin calls this between episodes."""
    if hasattr(model, "reset"):
        model.reset()

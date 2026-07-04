# WVWAM RoboTwin Success-Rate Adapter

This folder contains the RoboTwin policy adapter for online success-rate
evaluation.

## Install Into RoboTwin

From the RoboTwin repository root:

```bash
mkdir -p policy
ln -s /mnt/world_foundational_model/fyzhao/reverse_world_model/robotwin_success_rate/WVWAM policy/WVWAM
```

Copying also works if symlinks are inconvenient:

```bash
cp -r /mnt/world_foundational_model/fyzhao/reverse_world_model/robotwin_success_rate/WVWAM policy/WVWAM
```

## RoboTwin Config Fields

Set the policy import path:

```yaml
policy_name: WVWAM.deploy_policy
```

Add the WVWAM policy args from:

```text
/mnt/world_foundational_model/fyzhao/reverse_world_model/robotwin_success_rate/WVWAM/wvwam_policy_args.yml
```

The important fields are:

```yaml
wam_repo: /mnt/world_foundational_model/fyzhao/reverse_world_model
model_config_path: /mnt/world_foundational_model/fyzhao/reverse_world_model/validation_tmp/goal_reverse_joint.yml
checkpoint_path: /mnt/world_foundational_model/fyzhao/reverse_world_model/checkpoints/validation-goal-conditioned-joint-reverse-diffusion/checkpoint-1000
statistics_path: /mnt/damoxing/datasets/RoboTwin2_0_processed/robotwin_clean_5k_2/separate_statistics-state_dim128.json
action_only: false
reverse_world_order: true
action_dim: 14
action_type: qpos
camera_keys: head_camera,right_camera,left_camera
```

## Run

Use RoboTwin's normal evaluation command after selecting a task config, for
example:

```bash
cd /path/to/RoboTwin
python script/eval_policy.py --config path/to/your_wvwam_eval_config.yml
```

The exact task config path depends on the RoboTwin task you want to evaluate.

## What This Adapter Does

- Loads the WVWAM checkpoint through `pretrained_wam_path`.
- Reads RoboTwin online observations.
- Uses `head_camera`, `right_camera`, and `left_camera` RGB images.
- Uses `joint_action.vector` or qpos-like state fields as the robot state.
- Supports `original`, `goal_reverse_independent`, and `goal_reverse_joint`.
- Builds online video latents when `action_only=false`.
- Avoids double flipping when a Goal reverse model already returns forward actions.
- Denormalizes actions using the RoboTwin training statistics.
- Executes qpos actions through `TASK_ENV.take_action(action, action_type="qpos")`.

## Notes

Use `action_only=false` for the joint Goal variant. Set `action_only=true` only
for an explicit action-only ablation.

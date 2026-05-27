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

#!/usr/bin/env python
"""
Download pretrained models for Chain-of-Action.
"""
import os
import sys
import argparse
from huggingface_hub import snapshot_download
import glob

def download_model_for_task(task_name, save_dir="ckpt", repo_id="Solomonz/chain-of-action"):
    """Download pretrained model for a specific task. Always print the pt path if found, else empty string."""
    cache_dir = os.path.join(save_dir, "cache")
    os.makedirs(save_dir, exist_ok=True)

    pattern = f"*_{task_name}_*coa*.pt"
    pt_files = glob.glob(os.path.join(save_dir, "**", pattern), recursive=True)
    if pt_files:
        print(pt_files[0])
        return pt_files[0]
      

    # 没有则下载
    try:
        snapshot_download(
            cache_dir=cache_dir,
            local_dir=save_dir,
            repo_id=repo_id,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=[pattern],
        )
        pt_files = glob.glob(os.path.join(save_dir, "**", pattern), recursive=True)
        if pt_files:
            print(pt_files[0])
            return pt_files[0]
        else:
            print("No model found")
            return ""
    except Exception as e:
        print("Failed to download model")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Download pretrained models for Chain-of-Action")
    parser.add_argument("--task", type=str, default="push_button", help="Task name to download model for")
    parser.add_argument("--save_dir", type=str, default="coa", help="Directory to save models")
    parser.add_argument("--repo_id", type=str, default="Solomonz/chain-of-action", help="HuggingFace repository ID")
    args = parser.parse_args()
    download_model_for_task(args.task, args.save_dir, args.repo_id)

if __name__ == "__main__":
    main()

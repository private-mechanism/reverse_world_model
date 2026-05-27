#!/usr/bin/env python
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
Chain of Action (CoA) - Imitation Learning for Robot Manipulation
A framework for implementing and training Chain-of-Action models.
"""

from setuptools import setup, find_packages
import os

# Read the long description from README.md
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""



core_requirements = [
    "moviepy",
    "natsort",
    "omegaconf",
    "hydra-core",
    "hydra-joblib-launcher",
    "termcolor",
    "opencv-python-headless",
    "numpy<2",
    "imageio",
    "timm",
    "scipy",
    "einops",
    "diffusers",
    "accelerate",
    "clip @ git+https://github.com/openai/CLIP.git",
    "gymnasium @ git+https://git@github.com/stepjam/Gymnasium.git@0.29.2",
    "huggingface_hub",
    "moviepy==1.0.3",
    "wandb==0.13.8",
    "plotly==5.8.0",
    "open3d==0.19.0",
    "huggingface_hub",
    "transformers==4.33.3",
]
setup(  
    name="chain-of-action",
    version="0.1.0",
    author="Wenbo Zhang",
    author_email="zhang.wenbo@bytedance.com",
    description="A framework of Chain-of-Action for robot manipulation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://chain-of-action.github.io/",  
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    python_requires=">=3.8",
    # install_requires=read_requirements(),
    install_requires=core_requirements,
    
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "rlbench": [
            "rlbench @ git+https://git@github.com/stepjam/RLBench.git@b80e51feb3694d9959cb8c0408cd385001b01382",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "coa-train=scripts.train:main",
            "coa-eval=scripts.eval:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    
    keywords="robotics, imitation learning, chain of action, transformer, pytorch",
    
    project_urls={
        "Bug Reports": "https://github.com/zwbx/chain-of-action/issues",
        "Source": "https://github.com/zwbx/chain-of-action",
    },
    
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
        "cfgs": ["**/*.yaml", "**/*.yml"],
    },
) 
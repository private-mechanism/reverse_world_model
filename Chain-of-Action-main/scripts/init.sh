#!/bin/bash

# install system dependencies 
export DEBIAN_FRONTEND=noninteractive

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    kwin-wayland \
    kwin-wayland-backend-drm \
    kwin-wayland-backend-wayland \
    kwin-wayland-backend-x11 \
    weston \
    xserver-xephyr \
    xserver-xorg \
    xserver-xorg-legacy \
    xvfb \
    xwayland \
    libglu1-mesa \
    libxtst6 \
    libxv1 \
    wget \
    tmux \
    screen \
    libosmesa6-dev \
    patchelf \
    ncdu \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libgtk-3-0 \
    libxcomposite1 \
    libxrandr2 \
    xdg-utils

# set env variables
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

echo 'export COPPELIASIM_ROOT=${HOME}/CoppeliaSim' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT' >> ~/.bashrc
echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT' >> ~/.bashrc
source ~/.bashrc

wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# install chain-of-action and rlbench environment
pip install -e .[rlbench] 


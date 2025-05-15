#!/bin/sh

# Run this script to install Atari
pip install atari-py==0.2.9
pip install opencv-python==4.7.0.72
mkdir roms && cd roms
wget -L -nv http://www.atarimania.com/roms/Roms.rar
7z x -aoa Roms.rar
python -m atari_py.import_roms ROMS
cd .. && rm -rf roms

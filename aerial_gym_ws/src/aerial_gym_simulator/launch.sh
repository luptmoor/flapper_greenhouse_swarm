#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate aerialgym
gsettings set org.gnome.mutter check-alive-timeout 0
python3 flapper_greenhouse_swarm.py
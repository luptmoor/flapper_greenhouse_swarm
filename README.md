
# Installation
Clone with --recursive flag

Conda

```
conda create -n aerialgym python=3.8
conda activate aerialgym
```

Conda Dependencies
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
 pytorch-cuda=11.7 -c pytorch -c conda-forge -c defaults
conda install pyyaml==6.0 tensorboard==2.13.0 -c conda-forge -c pytorch -c defaults -c nvidia
# OR the newest version of PyTorch with CUDA version that supported by your driver if you like
# conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c pytorch3d pytorch3d
```

Install aerial gym
```
cd aerial_gym_ws/src/python
pip3 install -e .
# set the environment variables for Isaac Gym
export LD_LIBRARY_PATH=~/miniconda3/envs/aerialgym/lib
# OR
export LD_LIBRARY_PATH=~/anaconda3/envs/aerialgym/lib

# if you get an error message "rgbImage buffer error 999"
# then set this environment variable
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# Please add this your .bashrc or .zshrc
# file to avoid setting the environment variables
# every time you want to run the simulator
```

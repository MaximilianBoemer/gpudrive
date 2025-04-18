# GPUDrive


## Setup
```
apptainer shell --nv --bind /n/fs/pci-sharedt/mb9385:/mnt /n/fs/pci-sharedt/mb9385/gpudrive.sif
export PATH=/mnt/miniconda3/envs/gpudrive/bin:$PATH
source /u/mb9385/.bashrc
conda activate /mnt/miniconda3/envs/gpudrive
export MADRONA_MWGPU_KERNEL_CACHE=./gpudrive_cache
```

Setup scene generation as package
```
git submodule update --init --recursive
cd external/scene_generation
python setup.py install
cd scene_generation
pip install submodules/simple-knn
pip install submodules/diff-surfel-rasterization-mcmc
pip install submodules/diff-gaussian-rasterization-mcmc
pip install submodules/nvdiffrast
```

## Training agent
```
python baselines/ppo/ppo_sb3.py
```

## Run inference
```
python max_eval.py
```

## Visualization
Bird's eye view visualization:
```
max_viz.ipynb
```



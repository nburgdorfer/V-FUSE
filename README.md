# V-FUSE: Volumetric Depth Map Fusion with Long-Range Constraints
This is the official implementation of **V-FUSE: Volumetric Depth Map Fusion with Long-Range Constraints**.

[Nathaniel Burgdorfer](https://nburgdorfer.github.io),
[Philippos Mordohai](https://mordohai.github.io/)

### [Project page](https://nburgdorfer.github.io/vfuse/) | [Paper](https://arxiv.org/abs/2308.08715)

## Installation
### Conda Environment
For our environment setup, we use [conda](https://www.anaconda.com/download/). Please install conda and run the following command:
```bash
conda create -n vfuse python=3.9
```

Once created, activate the environment:
```bash
conda activate vfuse
```

### Python Dependancies
With the conda environment activated, install the python dependencies:
```bash
pip install -r requirements.txt --user
```

This project uses PyTorch, please install the latest version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Data Preparation
For DTU, we provide the input data used in our experiments from all four MVS methods:
- [MVSNet](https://stevens0-my.sharepoint.com/:u:/g/personal/nburgdor_stevens_edu/ESNvbUjv3UxBqvcUh2YHEDIBDIlVHDtJ-RxmGvjoJzTmRw?e=S24ML7)
- [UCSNet](https://stevens0-my.sharepoint.com/:u:/g/personal/nburgdor_stevens_edu/EdtU4wyHlvBJuEUJhf3EB0gBAnYV-FBv2zbp1jGdTawXig?e=dTdBpM)
- [NP-CVP-MVSNet](https://stevens0-my.sharepoint.com/:u:/g/personal/nburgdor_stevens_edu/EXUdw8TjMO9HojW95PgUnvcBXQgZMCdrH3gks_2ia562SA?e=QYU3zt)
- [GBi-Net]() (coming soon...)

For the Tanks & Temples intermediate set, we provide data for three of the MVS methods:
- [UCSNet]() (coming soon...)
- [NP-CVP-MVSNet]() (coming soon...)
- [GBi-Net]() (coming soon...)

For the Tanks & Temples advanced set, we provide data for one MVS method:
- [GBi-Net]() (coming soon...)

If you would like to use your own data, please follow the following format (using GBi-Net as example):
```
<gbinet>
  ->Cameras
    -> <scene000>
      ->00000000_cam.txt
      ->00000001_cam.txt
      ->00000002_cam.txt
      ->00000003_cam.txt
      ---
    -> <scene001>
    -> <scene002>
    ---
  ->Confs
    -> <scene000>
      ->00000000_conf.pfm
      ->00000001_conf.pfm
      ->00000002_conf.pfm
      ->00000003_conf.pfm
      ---
    -> <scene001>
    -> <scene002>
    ---
  ->Depths
    -> <scene000>
      ->00000000_depth.pfm
      ->00000001_depth.pfm
      ->00000002_depth.pfm
      ->00000003_depth.pfm
      ---
    -> <scene001>
    -> <scene002>
    ---
  ->Images
    -> <scene000>
      ->00000000.png
      ->00000001.png
      ->00000002.png
      ->00000003.png
      ---
    -> <scene001>
    -> <scene002>
    ---
```
If you would like to use your own data layout, please feel free to modify the code for the dataset in `src/datasets/<DATASET>.py`. NOTE: The `BaseDataset` class in the `src/dataset/BaseDataset.py` file should not need updating to support new datasets. Only the `build_dataset` function at the top of the file would need updating.

## Configuration
Before running training or inference, please update the config file for the dataset you would like to run. The config files are located under `configs/<DATASET>/` (e.g. `configs/DTU/dtu.yaml`). The only entries that need to be modified are the following:

```yaml
data_path: <path-to-dataset>
output_path: <desired-output-path>
model: <path-to-pretrained-model>

eval:
  data_path: <path-to-dtu-evaluation-data>
```
NOTE: The `eval:data_path` is a DTU specific config entry. This provides the path to the [DTU evaluation data](https://stevens0-my.sharepoint.com/:u:/g/personal/nburgdor_stevens_edu/EW69VFXgdVxHlfWDZdFGAjwB0OHjXUHOpAHDSAGVskq9yQ?e=qoAcSM).

## Training
To train our network from scratch, simply run the script:
```bash
./scripts/dtu_training.sh
```
A `./log/` folder will be created in the top-level directory of the repository, storing the current configuration as well as checkpoints for each epoch during training. If you would like to continue training from a specific checkpoint or pretrained model, edit the config file to include the path to the checkpoint:
```yaml
training:
  ckpt_file: <path-to-model-checkpoint>
```
Please leave this line blank if you do not wish to continue training from some model checkpoint.

## Inference
We provide [pretrained models](https://stevens0-my.sharepoint.com/:u:/g/personal/nburgdor_stevens_edu/EbhIvlrv1wNGkwjRXbZQANIBN2DGcdTTjL3_yKg0AjqXgg?e=Yf80J5) for each baseline MVS method.

To run inference on DTU/TNT/BlendedMVS, run the script:
```bash 
./scripts/<dtu|tnt|blendedmvs>_inference.sh
```
If you are running in a system with multiple GPU's, edit the line:
```bash
CUDA_VISIBLE_DEVICES=0
```
to the desired GPU ID #.

These bash scripts will run the `inference.py` script with the appropriate config file and dataset tag. The scenes that will be processed are the ones listed in the file `configs/<DATASET>/scene_lists/inference.txt`. The format for this file is one scene per line (case-sensitive). For the DTU dataset, the `inference.py` script automatically evaluates all point clouds output from our system following our dense evaluation script.

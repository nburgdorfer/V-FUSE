#!/bin/bash

CONFIG_PATH=configs/BlendedMVS/
DATASET=blendedmvs

CUDA_VISIBLE_DEVICES=0 python inference.py --config_path ${CONFIG_PATH} --dataset ${DATASET}

#!/bin/bash

CONFIG_PATH=configs/DTU/
DATASET=dtu

CUDA_VISIBLE_DEVICES=0 python training.py --config_path ${CONFIG_PATH} --dataset ${DATASET}

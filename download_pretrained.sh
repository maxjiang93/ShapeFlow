#!/bin/bash

echo "Downloading pretrained checkpoints and inference inputs. This will take a while, take a break and get a coffee..."

mkdir -p data
cd data
wget island.me.berkeley.edu/shape_flow/sparse_inputs.zip
unzip sparse_inputs.zip && rm sparse_inputs.zip
cd ..

mkdir -p runs
cd runs
wget island.me.berkeley.edu/shape_flow/pretrained_chair_symm128.zip
unzip pretrained_chair_symm128.zip && rm pretrained_chair_symm128.zip
cd ..

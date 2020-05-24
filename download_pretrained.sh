#!/bin/bash

mkdir -p data

echo "Downloading pretrained checkpoints and inference inputs. This will take a while, take a break and get a coffee..."

echo "Downloading Data..."
cd data
wget island.me.berkeley.edu/shape_flow/pretrained_chair_symm128.zip
wget island.me.berkeley.edu/shape_flow/sparse_inputs.zip

echo "Unzipping Data..."
unzip pretrained_chair_symm128.zip && rm pretrained_chair_symm128.zip
unzip sparse_inputs.zip && rm sparse_inputs.zip

cd ..

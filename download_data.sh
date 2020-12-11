#!/bin/bash

mkdir -p data

echo "Downloading and unzipping data. This will take a while, take a break and get a coffee..."

echo "Downloading Data..."

cd data
wget island.me.berkeley.edu/shape_flow/shapenet_simplified.zip
wget island.me.berkeley.edu/shape_flow/shapenet_thumbnails.zip
wget island.me.berkeley.edu/shape_flow/shapenet_pointcloud.zip
wget island.me.berkeley.edu/shape_flow/smpl_meshes.zip
wget island.me.berkeley.edu/shape_flow/sparse_inputs.zip
wget island.me.berkeley.edu/shape_flow/parametric_cad.zip

echo "Unzipping Data..."
unzip shapenet_simplified.zip && rm shapenet_simplified.zip
unzip shapenet_thumbnails.zip && rm shapenet_thumbnails.zip
unzip shapenet_pointcloud.zip && rm shapenet_pointcloud.zip
unzip smpl_meshes.zip && rm smpl_meshes.zip
unzip sparse_inputs.zip && rm sparse_inputs.zip
unzip parametric_cad.zip && rm parametric_cad.zip

cd ..

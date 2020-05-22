#!/bin/bash

mkdir -p data

echo "Downloading and unzipping data. This will take a while, take a break and get a coffee..."

echo "Downloading Data..."
wget island.me.berkeley.edu/datasets/shapenet_pointcloud.zip .
wget island.me.berkeley.edu/datasets/shapenet_simplified.zip .
wget island.me.berkeley.edu/datasets/shapenet_thumbnails.zip .

mv shapenet_pointcloud.zip shapenet_simplified.zip shapenet_thumbnails.zip data
cd data
echo "Unzipping Data..."
unzip shapenet_pointcloud.zip
unzip shapenet_simplified.zip
unzip shapenet_thumbnails.zip

rm *.zip
cd ..

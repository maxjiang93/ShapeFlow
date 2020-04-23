#!/bin/bash

mkdir -p data

echo "Downloading and unzipping data. This will take a while, take a break and get a coffee..."

echo "Downloading Data..."
wget island.me.berkeley.edu/datasets/shapenet_chair_watertight.zip .
wget island.me.berkeley.edu/datasets/shapenet_chair_simplified.zip .
wget island.me.berkeley.edu/datasets/shapenet_chair_thumbnails.zip .

mv shapenet_chair_watertight.zip shapenet_chair_simplified.zip shapenet_chair_thumbnails.zip data
cd data
echo "Unzipping Data..."
unzip shapenet_chair_watertight.zip
unzip shapenet_chair_simplified.zip
unzip shapenet_chair_thumbnails.zip

rm *.zip
cd ..

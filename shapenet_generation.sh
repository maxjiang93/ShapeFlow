#!/bin/bash

# CONFIG
category=02691156
ckpt=runs/pretrained_airplane_symm128/checkpoint_latest.pth.tar_deepdeform_200.pth.tar

gpuid=$1  # gpu id
sid=$2  # start id
eid=$3  # end id

export CUDA_VISIBLE_DEVICES=$gpuid

allfiles=($(ls data/sparse_inputs/$category/*.ply))

for i in $(seq $sid $((eid-1))); do 
    infile=${allfiles[$i]}
    python shapenet_reconstruct.py --input_path=$infile --output_dir=out --checkpoint=$ckpt --device=cuda
done

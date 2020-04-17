#!/bin/bash

run_name=run_0
log_dir=runs/$run_name
data_root=/tmp/data/shapenet_watertight

module load pytorch/v1.4.0-gpu
mkdir -p runs

# export CUDA_VISIBLE_DEVICES=0

srun python shapenet_train.py \
--adjoint \
--data_root=$data_root \
--batch_size_per_gpu=32 \
--pseudo_eval_epoch_size=128 \
--pseudo_train_epoch_size=2048 \
--epochs=100 \
--lr=1e-3 \
--log_dir=$log_dir \
--nsamples=2048 \
--lat_dims=64 \
--encoder_nf=32 \
--deformer_nf=64 \
--lr_scheduler \
--no_normals \
--visualize_mesh \

#!/bin/bash

run_name=run_debug
log_dir=runs/$run_name
data_root=data/shapenet_watertight

module load pytorch/v1.4.0-gpu
mkdir -p runs

# export CUDA_VISIBLE_DEVICES=0

srun python shapenet_train.py \
--solver='dopri5' \
--atol=1e-4 \
--rtol=1e-4 \
--data_root=$data_root \
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
--no_pn_batchnorm \
--batch_size_per_gpu=1 \
--log_interval=1 \
--debug \
--no_adjoint \

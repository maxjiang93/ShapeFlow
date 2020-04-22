#!/bin/bash

run_name=run_debug
log_dir=runs/$run_name
data_root=data/shapenet_watertight

module load pytorch/v1.4.0-gpu
mkdir -p runs

# export CUDA_VISIBLE_DEVICES=0

srun python -m pdb shapenet_train.py \
--atol=1e-4 \
--rtol=1e-4 \
--data_root=$data_root \
--pseudo_eval_epoch_size=128 \
--lr=1e-3 \
--log_dir=$log_dir \
--encoder_nf=32 \
--lr_scheduler \
--no_normals \
--visualize_mesh \
--pn_batchnorm \
--batch_size_per_gpu=3 \
--log_interval=1 \
--debug \
--adjoint \
--epochs=1 \
--solver='dopri5' \
--deformer_nf=100 \
--pseudo_train_epoch_size=1000 \
--nsamples=5000 \
--lat_dims=32 \

#!/bin/bash

run_name=run_encx0.1_18obj_dissim
log_dir=runs/$run_name
data_root=data/shapenet_watertight

# module load pytorch/v1.4.0-gpu
mkdir -p runs

export CUDA_VISIBLE_DEVICES=1,2

python shapenet_train.py \
--atol=1e-4 \
--rtol=1e-4 \
--data_root=$data_root \
--pseudo_train_epoch_size=2048 \
--pseudo_eval_epoch_size=128 \
--lr=1e-3 \
--log_dir=$log_dir \
--lr_scheduler \
--no_normals \
--visualize_mesh \
--pn_batchnorm \
--batch_size_per_gpu=6 \
--log_interval=10 \
--no_adjoint \
--epochs=100 \
--solver='rk4' \
--deformer_nf=128 \
--nsamples=2048 \
--lat_dims=64 \
--encoder_nf=32 \
--dropout_prob=0.0 \
--nonlin='leakyrelu' \
# --resume='runs/run_1/checkpoint_latest.pth.tar_deepdeform_036.pth.tar' \


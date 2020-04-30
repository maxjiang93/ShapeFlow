#!/bin/bash

run_name=run_hub_spoke_4746
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
--visualize_mesh \
--batch_size_per_gpu=24 \
--log_interval=10 \
--adjoint \
--epochs=101 \
--solver='dopri5' \
--deformer_nf=128 \
--nsamples=512 \
--lat_dims=128 \
--nonlin='leakyrelu' \
# --resume='runs/run_encless/checkpoint_latest.pth.tar_deepdeform_100.pth.tar'


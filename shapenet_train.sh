#!/bin/bash

run_name=run_4746_allnr_nosign
log_dir=runs/$run_name
data_root=data/shapenet_simplified

module load pytorch/v1.4.0-gpu
mkdir -p runs

# export CUDA_VISIBLE_DEVICES=0

srun python shapenet_train.py \
--atol=1e-4 \
--rtol=1e-4 \
--data_root=$data_root \
--pseudo_train_epoch_size=2048 \
--pseudo_eval_epoch_size=128 \
--lr=1e-3 \
--log_dir=$log_dir \
--lr_scheduler \
--visualize_mesh \
--batch_size_per_gpu=128 \
--log_interval=2 \
--epochs=100 \
--no_sign_net \
--adjoint \
--solver='dopri5' \
--deformer_nf=128 \
--nsamples=512 \
--lat_dims=512 \
--nonlin='leakyrelu' \
--sampling_method='all_no_replace' \
# --resume='runs/run_hub_spoke_all_chair/checkpoint_latest.pth.tar_deepdeform_085.pth.tar'


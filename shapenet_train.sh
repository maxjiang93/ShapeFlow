#!/bin/bash

#####################################
#       Configure Experiment        #
#####################################
run_name=demo
log_dir=runs/$run_name
data_root=data/shapenet_simplified

# Create run directory if it doesn't exist.
mkdir -p runs

# Launch training.
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
--batch_size_per_gpu=32 \
--log_interval=2 \
--epochs=100 \
--no_sign_net \
--adjoint \
--solver='dopri5' \
--deformer_nf=128 \
--nsamples=512 \
--lat_dims=128 \
--nonlin='leakyrelu' \
--symm \
--category=${category} \
--sampling_method='all_no_replace' \

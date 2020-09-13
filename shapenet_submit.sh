#!/bin/bash
#SBATCH -o .slurm_logs/symm_lat128_nosign_b32.out
#SBATCH -C gpu
#SBATCH -t 480
#SBATCH -c 80
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH -A dasrepo
#SBATCH --requeue
#SBATCH -J pytorch

module load pytorch/v1.4.0-gpu
mkdir -p logs

# # move data to SSD
# cp -r data /tmp

run_name=symm_lat128_nosign_b32
log_dir=runs/$run_name
data_root=data/shapenet_simplified

# run
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
--sampling_method='all_no_replace'
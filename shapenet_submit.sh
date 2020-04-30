#!/bin/bash
#SBATCH -o .slurm_logs/run_hs_lat512_bs128_nodyns.out
#SBATCH -C gpu
#SBATCH -t 240
#SBATCH -c 80
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -A dasrepo
#SBATCH --requeue
#SBATCH -J pytorch

module load pytorch/v1.4.0-gpu
mkdir -p logs

# # move data to SSD
# cp -r data /tmp

run_name=run_hs_lat512_bs128_nodyns
log_dir=runs/$run_name
data_root=data/shapenet_simplified

# run
srun -l python shapenet_train.py \
--atol=1e-4 \
--rtol=1e-4 \
--data_root=$data_root \
--pseudo_train_epoch_size=2048 \
--pseudo_eval_epoch_size=128 \
--lr=1e-3 \
--log_dir=$log_dir \
--lr_scheduler \
--visualize_mesh \
--batch_size_per_gpu=64 \
--log_interval=2 \
--adjoint \
--epochs=101 \
--solver='dopri5' \
--deformer_nf=128 \
--nsamples=512 \
--lat_dims=512 \
--nonlin='leakyrelu' \
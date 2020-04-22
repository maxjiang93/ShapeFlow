#!/bin/bash
#SBATCH -o .slurm_logs/run_3.out
#SBATCH -C gpu
#SBATCH -t 240
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

run_name=run_3
log_dir=runs/$run_name
data_root=data/shapenet_watertight

# run
srun -l python shapenet_train.py \
--adjoint \
--solver='dopri5' \
--atol=1e-4 \
--rtol=1e-4 \
--data_root=$data_root \
--batch_size_per_gpu=8 \
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
--log_interval=1 \
--no_pn_batchnorm \
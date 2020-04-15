#!/bin/bash
#SBATCH -o .slurm_logs/run_0.out
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

run_name=run_0
log_dir=logs/$run_name
data_root=../../data/shapenet

# run
srun -l python train_shapenet.py \
--batch_size_per_gpu=16 \
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

#!/bin/bash
#SBATCH --job-name="Brucknem - VGG16 - 10k - Random"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python3 ../module.py -c -v -vv -t --save_every_epoch 1 --num_epochs 1000 -m torch_vgg16 --subset_size 10000 -d vgg16_10k_r -r

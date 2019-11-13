#!/bin/bash
#SBATCH --job-name="Brucknem - VGG16 - 500"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python3 module.py -c -t -v -vv -m vgg16 --num_epochs=500

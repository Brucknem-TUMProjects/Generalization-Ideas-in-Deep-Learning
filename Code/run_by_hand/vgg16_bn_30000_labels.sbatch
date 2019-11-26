#!/bin/bash
#SBATCH --job-name="Brucknem - vgg16_bn - 30000 - labels"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:6G
#SBATCH --mem=6G
#SBATCH --time=3-23:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

srun python3 ../module.py -c -ed -t -v -vv -d /storage/slurm/brucknem/vgg16_bn_30000_labels -ss 30000 -nth 50 -st100 -sl 10 -bs 16

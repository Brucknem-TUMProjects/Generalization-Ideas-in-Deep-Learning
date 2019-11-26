#!/bin/bash
#SBATCH --job-name="Brucknem - vgg16_bn - confusion - 1000 "
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:6G
#SBATCH --mem=6G
#SBATCH --time=10-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

srun python3 ../module.py -c -ed -t -v -vv -d /storage/slurm/brucknem/vgg16_bn_confusion_1000 -ss 10000 -nth 50 -st100 -sl 10 -bs 16 -cs 1000

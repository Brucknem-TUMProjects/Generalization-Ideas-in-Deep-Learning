#!/bin/bash
#SBATCH --job-name="Brucknem - vgg16_bn - confusion - 5000"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=12G
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

srun python3 ../module.py -c -ed -t -v -vv -d /storage/slurm/brucknem/vgg16_bn_confusion_5000 -ss 10000 -nth 50 -st100 -sl 10 -bs 16 -cs 5000

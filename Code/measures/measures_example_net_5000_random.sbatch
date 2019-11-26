#!/bin/bash
#SBATCH --job-name="Brucknem - Measures - ExampleNet - 5000 - Random"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python3 ../measures.py -d /storage/slurm/brucknem/ExampleNet_5000_random

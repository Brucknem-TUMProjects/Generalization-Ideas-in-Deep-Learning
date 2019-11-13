#!/bin/bash
#SBATCH --job-name="Brucknem - Small test"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
dt=`date '+%d-%m-%Y_%H-%M-%S'` 
srun python3 module.py -c -t -v -vv -f $dt


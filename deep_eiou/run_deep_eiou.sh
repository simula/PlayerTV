#!/bin/sh
#SBATCH --job-name=create_hota
#SBATCH --output=output%j.txt
#SBATCH --error=testpython%j.err
#SBATCH --partition=dgx2q
#SBATCH --ntasks=1
#SBATCH --time=9:50:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1

source /cm/shared/apps/anaconda3/x86_64/2022.05/etc/profile.d/conda.sh
conda activate DeepEIoU
python player_tv.py
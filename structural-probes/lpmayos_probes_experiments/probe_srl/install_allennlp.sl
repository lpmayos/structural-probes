#!/bin/bash
#SBATCH --job-name="inst"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=10Gb
#SBATCH -p high

source /homedtic/lperez/miniconda3/etc/profile.d/conda.sh
conda activate /homedtic/lperez/miniconda3/envs/env36
pip install allennlp
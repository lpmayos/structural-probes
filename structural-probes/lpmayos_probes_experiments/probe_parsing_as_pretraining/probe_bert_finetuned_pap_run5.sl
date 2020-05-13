#!/bin/bash
#SBATCH --job-name="pr_pap5"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=30Gb
#SBATCH -p high
#SBATCH --gres=gpu:1

module load Python/3.6.4-foss-2017a
module load cuDNN/7.6.3.30-CUDA-10.0.130

source /homedtic/lperez/miniconda3/etc/profile.d/conda.sh
conda activate /homedtic/lperez/miniconda3/envs/persing_env

cd /homedtic/lperez/structural-probes/structural-probes/

python probe_model_checkpoints_pap.py --probe_name naacl_19_ptb \
                                      --ptb_path /homedtic/lperez/datasets/PTB_SD_3_3_0/ \
                                      --probes_input_paths /homedtic/lperez/datasets/PTB_SD_3_3_0/train.gold.txt /homedtic/lperez/datasets/PTB_SD_3_3_0/dev.gold.txt /homedtic/lperez/datasets/PTB_SD_3_3_0/test.gold.txt \
                                      --parse_distance_yaml /homedtic/lperez/structural-probes/example/config/naacl19/bert-base/ptb-pad-BERTbase7.yaml \
                                      --parse_depth_yaml /homedtic/lperez/structural-probes/example/config/naacl19/bert-base/ptb-prd-BERTbase7.yaml \
                                      --checkpoints_path /homedtic/lperez/parsing-as-pretraining/runs_constituency_parsing/run5/output \
                                      --seed 50 \
                                      --model_partial_name pytorch_model

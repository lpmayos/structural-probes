#!/bin/bash
#SBATCH --job-name="pr_mul1"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=30Gb
#SBATCH -p high
#SBATCH --gres=gpu:1

module load Python/3.6.4-foss-2017a
module load cuDNN/7.6.3.30-CUDA-10.0.130

source /homedtic/lperez/miniconda3/etc/profile.d/conda.sh
conda activate /homedtic/lperez/miniconda3/envs/env36

cd /homedtic/lperez/structural-probes/structural-probes/

python probe_model_checkpoints.py --probe_name naacl_19_ptb \
                                  --ptb_path /homedtic/lperez/datasets/PTB_SD_3_3_0/ \
                                  --probes_input_paths /homedtic/lperez/datasets/PTB_SD_3_3_0/train.gold.txt /homedtic/lperez/datasets/PTB_SD_3_3_0/dev.gold.txt /homedtic/lperez/datasets/PTB_SD_3_3_0/test.gold.txt \
                                  --parse_distance_yaml /homedtic/lperez/structural-probes/example/config/naacl19/bert-base/ptb-pad-BERTbase7.yaml \
                                  --parse_depth_yaml /homedtic/lperez/structural-probes/example/config/naacl19/bert-base/ptb-prd-BERTbase7.yaml \
                                  --models_path /homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_parsing_multilingual/run1 \
                                  --checkpoints_path /results_parsing \
                                  --checkpoints 250 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 3750 4000 4250 4500 4750 5000 5250 5500 5750 6000 6250 6500 6750 7000 7250 7500 7750 \
                                  --seed 10

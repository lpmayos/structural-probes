#!/bin/bash
#SBATCH --job-name="run5"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=30Gb
#SBATCH -p high
#SBATCH --gres=gpu:1

module load Python/3.6.4-foss-2017a
module load cuDNN/7.6.3.30-CUDA-10.0.130

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate env36
which python

python bert_base_cased_finetuned_squad.py --probe_name naacl_19_ptb \
                                          --ptb_path /homedtic/lperez/datasets/PTB_SD_3_3_0/ \
                                          --probes_input_paths /homedtic/lperez/datasets/PTB_SD_3_3_0/train.gold.txt /homedtic/lperez/datasets/PTB_SD_3_3_0/dev.gold.txt /homedtic/lperez/datasets/PTB_SD_3_3_0/test.gold.txt \
                                          --parse_distance_yaml /homedtic/lperez/structural-probes/example/config/naacl19/bert-base/ptb-pad-BERTbase7.yaml \
                                          --parse_depth_yaml /homedtic/lperez/structural-probes/example/config/naacl19/bert-base/ptb-prd-BERTbase7.yaml \
                                          --models_path /homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_squad/run5 \
                                          --checkpoints 2000 3750 7250 9000 12750 14500 18250 20000 \
                                          --squad_dataset_file /homedtic/lperez/transformers/examples/tests_samples/SQUAD/dev-v1.1.json \

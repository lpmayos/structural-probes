#!/bin/bash
#SBATCH --job-name="run1"
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
                                          --ptb_path /home/lpmayos/hd/code/datasets/PTB_SD_3_3_0/ \
                                          --probes_input_paths /home/lpmayos/hd/code/datasets/PTB_SD_3_3_0_mini/train.gold.txt /home/lpmayos/hd/code/datasets/PTB_SD_3_3_0_mini/dev.gold.txt /home/lpmayos/hd/code/datasets/PTB_SD_3_3_0_mini/test.gold.txt \
                                          --parse_distance_yaml /home/lpmayos/hd/code/structural-probes/example/config/naacl19/bert-base/ptb-pad-BERTbase7.yaml \
                                          --parse_depth_yaml /home/lpmayos/hd/code/structural-probes/example/config/naacl19/bert-base/ptb-prd-BERTbase7.yaml \
                                          --models_path /home/lpmayos/hd/code/transformers/lpmayos_experiments/bert_base_cased_finetuned_squad/run1 \
                                          --checkpoints 250 5500 11000 16500 22000 \
                                          --squad_dataset_file /home/lpmayos/hd/code/transformers/examples/tests_samples/SQUAD/dev-v1.1.json \








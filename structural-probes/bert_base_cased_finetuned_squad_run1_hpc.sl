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
                                          --ptb_path /homedtic/lperez/datasets/PTB_SD_3_3_0/ \
                                          --probes_input_paths /homedtic/lperez/datasets/PTB_SD_3_3_0/train.gold.txt /homedtic/lperez/datasets/PTB_SD_3_3_0/dev.gold.txt /homedtic/lperez/datasets/PTB_SD_3_3_0/test.gold.txt \
                                          --parse_distance_yaml /homedtic/lperez/structural-probes/example/config/naacl19/bert-base/ptb-pad-BERTbase7.yaml \
                                          --parse_depth_yaml /homedtic/lperez/structural-probes/example/config/naacl19/bert-base/ptb-prd-BERTbase7.yaml \
                                          --models_path /homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_squad/run1 \
                                          --checkpoints 0 250 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 3750 4000 4250 4500 4750 5000 5250 5500 5750 6000 6250 6500 6750 7000 7250 7500 7750 8000 8250 8500 8750 9000 9250 9500 9750 10000 10250 10500 10750 11000 11250 11500 11750 12000 12250 12500 12750 13000 13250 13500 13750 14000 14250 14500 14750 15000 15250 15500 15750 16000 16250 16500 16750 17000 17250 17500 17750 18000 18250 18500 18750 19000 19250 19500 19750 20000 20250 20500 20750 21000 21250 21500 21750 22000 \
                                          --squad_dataset_file /homedtic/lperez/transformers/examples/tests_samples/SQUAD/dev-v1.1.json \

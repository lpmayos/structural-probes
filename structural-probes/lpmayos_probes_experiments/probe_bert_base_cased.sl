#!/bin/bash
#SBATCH --job-name="pr_bert0"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=30Gb
#SBATCH -p high
#SBATCH --gres=gpu:1

module load Python/3.6.4-foss-2017a
module load cuDNN/7.6.3.30-CUDA-10.0.130

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate env36

SEED=10
OUTPUT_DIR=/homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_pos/run0

if [[ ! -d "$OUTPUT_DIR" ]]
then
    echo "$OUTPUT_DIR does not exist; we create it now."
    mkdir $OUTPUT_DIR
fi

if [[ ! -d "$OUTPUT_DIR/results" ]]
then
    echo "$OUTPUT_DIR/results does not exist; we create it now."
    mkdir $OUTPUT_DIR/results
fi

if [[ ! -d "$OUTPUT_DIR/results/checkpoint-0" ]]
then
    echo "$OUTPUT_DIR/results/checkpoint-0 does not exist; we create it now."
    mkdir $OUTPUT_DIR/results/checkpoint-0
fi


cd /homedtic/lperez/structural-probes/structural-probes/

python probe_model_checkpoints.py --probe_name naacl_19_ptb \
                                  --ptb_path /homedtic/lperez/datasets/PTB_SD_3_3_0/ \
                                  --probes_input_paths /homedtic/lperez/datasets/PTB_SD_3_3_0/train.gold.txt /homedtic/lperez/datasets/PTB_SD_3_3_0/dev.gold.txt /homedtic/lperez/datasets/PTB_SD_3_3_0/test.gold.txt \
                                  --parse_distance_yaml /homedtic/lperez/structural-probes/example/config/naacl19/bert-base/ptb-pad-BERTbase7.yaml \
                                  --parse_depth_yaml /homedtic/lperez/structural-probes/example/config/naacl19/bert-base/ptb-prd-BERTbase7.yaml \
                                  --models_path $OUTPUT_DIR \
                                  --checkpoints_path /results \
                                  --checkpoints 0 \
				  --model_to_load bert-base-cased \
				  --seed $SEED

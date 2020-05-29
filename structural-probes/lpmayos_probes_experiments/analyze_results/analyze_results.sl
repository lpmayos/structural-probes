
# Copy bert-base-cased probes to checkpoints-0

analyze_results=0

if [ "$analyze_results" == 1 ]; then
    python copy_probes_checkpoints_zero.py --models_path /homedtic/lperez/transformers/lpmayos_experiments \
                                           --probes_checkpoint_zero_path /homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_squad/run0/results/checkpoint-0/structural_probes
fi

# SQUAD

analyze_results=0

if [ "$analyze_results" == 1 ]; then
    echo "Analyzing SQuAD results"
    python bert_base_cased_finetuned_squad_analyse_results.py --probe_name naacl_19_ptb \
                                                              --models_path /homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_squad \
                                                              --runs run1 run2 run3 run4 run5 \
                                                              --checkpoints 0 250 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 3750 4000 4250 4500 4750 5000 5250 5500 5750 6000 6250 6500 6750 7000 7250 7500 7750 8000 8250 8500 8750 9000 9250 9500 9750 10000 10250 10500 10750 11000 11250 11500 11750 12000 12250 12500 12750 13000 13250 13500 13750 14000 14250 14500 14750 15000 15250 15500 15750 16000 16250 16500 16750 17000 17250 17500 17750 18000 18250 18500 18750 19000 19250 19500 19750 20000 20250 20500 20750 21000 21250 21500 21750 22000 \
                                                              --output_file bert_base_cased_finetuned_squad_results.json
fi

# GLUE

analyze_results=0

if [ "$analyze_results" == 1 ]; then
    echo "Analyzing GLUE results"
    python bert_base_cased_finetuned_glue_analyse_results.py --probe_name naacl_19_ptb \
                                                              --models_path /homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_glue \
                                                              --runs run1 run2 run3 run4 run5 \
                                                              --output_file bert_base_cased_finetuned_mrpc_results.json \
                                                              --task_name MRPC

    python bert_base_cased_finetuned_glue_analyse_results.py --probe_name naacl_19_ptb \
                                                              --models_path /homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_glue \
                                                              --runs run1 run2 run3 run4 run5 \
                                                              --output_file bert_base_cased_finetuned_qqp_results.json \
                                                              --task_name QQP

fi

# POS Tagging

analyze_results=0

if [ "$analyze_results" == 1 ]; then
    echo "Analyzing POS Tagging results"
    python bert_base_cased_finetuned_pos_analyse_results.py --probe_name naacl_19_ptb \
                                                            --models_path /homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_pos \
                                                            --runs run1 run2 run3 run4 run5 \
                                                            --output_file bert_base_cased_finetuned_pos_results.json
fi

# Parsing

analyze_results=0

if [ "$analyze_results" == 1 ]; then
    echo "Analyzing parsing results"
    python bert_base_cased_finetuned_parsing_analyse_results.py --probe_name naacl_19_ptb \
                                                                --models_path /homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_parsing \
                                                                --runs run1 run2 run3 run4 run5 \
                                                                --output_file bert_base_cased_finetuned_parsing_results.json
fi


# SRL

analyze_results=0

if [ "$analyze_results" == 1 ]; then
    echo "Analyzing SRL results"
    python bert_base_cased_finetuned_srl_analyse_results.py --probe_name naacl_19_ptb \
                                                            --models_path /homedtic/lperez/bert_finetuned_srl \
                                                            --runs run1 run2 run3 run4 run5 \
                                                            --output_file bert_base_cased_finetuned_srl_results.json
fi



# Parsing as Pretraining: Constituent Parsing

analyze_results=0

if [ "$analyze_results" == 1 ]; then
    echo "Analyzing PAP Constituents results"
    python bert_base_cased_finetuned_pap_analyse_results.py --probe_name naacl_19_ptb \
                                                            --models_path /homedtic/lperez/parsing-as-pretraining/runs_constituency_parsing \
                                                            --runs run1 run2 run3 run4 run5 \
                                                            --output_file bert_base_cased_finetuned_pap_constituents_results.json
fi



# Parsing multilingual

analyze_results=0

if [ "$analyze_results" == 1 ]; then
    echo "Analyzing parsing results"
    python bert_base_cased_finetuned_parsing_analyse_results.py --probe_name naacl_19_ptb \
                                                                --models_path /homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_parsing_multilingual \
                                                                --runs run1 run2 run3 run4 run5 \
                                                                --output_file bert_base_cased_finetuned_parsing_multilingual_results.json
fi




# Parsing ptb sd

analyze_results=0

if [ "$analyze_results" == 1 ]; then
    echo "Analyzing parsing results"
    python bert_base_cased_finetuned_parsing_analyse_results.py --probe_name naacl_19_ptb \
                                                                --models_path /homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_parsing_ptb_sd \
                                                                --runs run1 run2 run3 run4 run5 \
                                                                --output_file bert_base_cased_finetuned_parsing_ptb_results.json
fi

analyze_mlm=1

if [ "$analyze_mlm" == 1 ]; then
    echo "Analyzing mlm results"
    python bert_base_cased_finetuned_parsing_add_mlm_perplexities.py --models_path /homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_parsing_ptb_sd \
                                                                     --output_file bert_base_cased_finetuned_parsing_ptb_results.json
fi

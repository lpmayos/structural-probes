from __future__ import print_function

import os
import re
from argparse import ArgumentParser
import json
import logging


def analyse_results(probe_name, models_path, runs_list, checkpoints_list, output_file):
    """
    """

    results = {}

    for run in runs_list:

        run_path = models_path + '/' + run
        results[run] = {}

        for checkpoint in checkpoints_list:

            logging.info('Reading results for checkpoint %s' % checkpoint)

            checkpoint_path = run_path + '/results/checkpoint-' + checkpoint
            squad_results_path = checkpoint_path + '/squad_results.json'
            mlm_results_path = checkpoint_path + '/eval_mlm_results.txt'
            probes_path = checkpoint_path + '/structural_probes/' + probe_name

            checkpoint_results = {'squad_exact_match': None,
                                  'squad_f1': None,
                                  'mlm_preplexity': None,
                                  'parse-depth': {},
                                  'parse-distance': {}}

            if os.path.exists(squad_results_path):
                with open(squad_results_path) as file:
                    data = json.load(file)
                    checkpoint_results['squad_exact_match'] = data['squad_exact_match']
                    checkpoint_results['squad_f1'] = data['squad_f1']
            else:
                logging.info('File %s does not exists yet' % squad_results_path)

            if os.path.exists(mlm_results_path):
                with open(mlm_results_path) as file:
                    line = file.readline()
                    pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+")  # regular expression to match floats, with optional +/-
                    mlm_preplexity = float(pattern.findall(line)[0])
                    checkpoint_results['mlm_preplexity'] = mlm_preplexity
            else:
                logging.info('File %s does not exists yet' % mlm_results_path)

            dev_root_acc_file = probes_path + '/parse-depth/dev.root_acc'
            if os.path.exists(dev_root_acc_file):
                with open(dev_root_acc_file) as file:
                    result = file.readlines()[0].split()[0]  # i.e. 0.8123529411764706      1381    1700
                    checkpoint_results['parse-depth']['dev.root_acc'] = float(result)
            else:
                logging.info('File %s does not exists yet' % dev_root_acc_file)

            dev_spearmanr_file = probes_path + '/parse-depth/dev.spearmanr-5_50-mean'
            if os.path.exists(dev_spearmanr_file):
                with open(dev_spearmanr_file) as file:
                    result = file.readlines()[0]  # i.e. 0.8552411954331226
                    checkpoint_results['parse-depth']['dev.spearmanr-5_50-mean'] = float(result)
            else:
                logging.info('File %s does not exists yet' % dev_spearmanr_file)

            dev_uuas_file = probes_path + '/parse-distance/dev.uuas'
            if os.path.exists(dev_uuas_file):
                with open(dev_uuas_file) as file:
                    result = file.readlines()[0]  # i.e. 0.7040907201804905
                    checkpoint_results['parse-distance']['dev.uuas'] = float(result)
            else:
                logging.info('File %s does not exists yet' % dev_uuas_file)

            dev_spearman_file = probes_path + '/parse-distance/dev.spearmanr-5_50-mean'
            if os.path.exists(dev_spearman_file):
                with open(dev_spearman_file) as file:
                    result = file.readlines()[0]  # i.e. 0.8058858201615192
                    checkpoint_results['parse-distance']['dev.spearmanr-5_50-mean'] = float(result)
            else:
                logging.info('File %s does not exists yet' % dev_spearman_file)

    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4, sort_keys=True)


if __name__ == '__main__':
    """ python bert_base_cased_finetuned_squad_analyse_results.py --probe_name naacl_19_ptb --models_path /homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_squad --runs run1 run2 run3 run4 run5 --checkpoints 250 2000 3750 5500 7250 9000 11000 12750 14500 16500 18250 20000 22000 --output_file bert_base_cased_finetuned_squad_results.json
    """

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    parser = ArgumentParser()

    parser.add_argument("--probe_name", default=None, type=str, required=True, help="naacl_19_ptb")
    parser.add_argument("--models_path", default=None, type=str, required=True, help="/home/lpmayos/hd/code/transformers/lpmayos_experiments/bert_base_cased_finetuned_squad")
    parser.add_argument('--runs', nargs='+', required=True, help="run1 run2 run3 run4 run5")
    parser.add_argument('--checkpoints', nargs='+', required=True, help="250 2000 3750 5500 7250 9000 11000 12750 14500 16500 18250 20000 22000")
    parser.add_argument("--output_file", default=None, type=str, required=True, help="bert_base_cased_finetuned_squad_results.json")

    args = parser.parse_args()
    analyse_results(args.probe_name, args.models_path, args.runs, args.checkpoints, args.output_file)

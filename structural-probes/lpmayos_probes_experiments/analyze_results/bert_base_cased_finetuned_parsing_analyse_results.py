from __future__ import print_function

import os
from argparse import ArgumentParser
import json
import logging


def analyse_results(probe_name, models_path, runs_list, output_file):
    """
    """

    results = {}

    for run in runs_list:

        run_results_path = models_path + '/' + run + '/results_parsing'

        results_path = run_results_path + '/eval_results.txt'
        parsing_results = {}
        with open(results_path) as file:
            for line in file.readlines():
                if not line.startswith('./results_parsing'):
                    parts = line.split(' = ')
                    checkpoint = parts[0].split('_')[0]
                    metric = parts[0].split('_')[1]
                    metric_result = float(parts[1].replace('\n', ''))
                    if not checkpoint in parsing_results:
                        parsing_results[checkpoint] = {}
                    parsing_results[checkpoint][metric] = metric_result

        # add checkpoint-0 results
        parsing_results['0'] = {}
        checkpoint_zero_results_path =  models_path + '/' + run + '/results_parsing/checkpoint-0/test_results.txt'
        with open(checkpoint_zero_results_path) as file:
            for line in file.readlines():
                parts = line.split(' = ')
                metric = parts[0]
                metric_result = float(parts[1].replace('\n', ''))
                parsing_results['0'][metric] = metric_result

        results[run] = {}

        checkpoints_list = [os.path.join(run_results_path, o) for o in os.listdir(run_results_path) if os.path.isdir(os.path.join(run_results_path, o)) and o.startswith(('checkpoint-'))]

        for checkpoint_path in checkpoints_list:

            checkpoint = checkpoint_path.split('/')[-1].split('-')[-1]

            logging.info('Reading results for checkpoint %s' % checkpoint)

            probes_path = checkpoint_path + '/structural_probes/' + probe_name
            checkpoint_results = {'uas': parsing_results[checkpoint]['uas'],
                                  'las': parsing_results[checkpoint]['las'],
                                  'loss': parsing_results[checkpoint]['loss'],
                                  'label accuracy score': parsing_results[checkpoint]['label accuracy score'],
                                  'parse-depth': {
                                      'dev.root_acc': None,
                                      'dev.spearmanr-5_50-mean': None
                                  },
                                  'parse-distance': {
                                      'dev.spearmanr-5_50-mean': None,
                                      'dev.uuas': None
                                  }}

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

            results[run][int(checkpoint)] = checkpoint_results

        # add checkpoint-0 probes results
        # We simply copy the results from another task (i.e. parsing UD EN EWT)

        with open('bert_base_cased_finetuned_parsing_results.json') as json_file:
            parsing_results_ud = json.load(json_file)

        parsing_results[run]['0']['parse-depth'] = parsing_results_ud['run1']['0']['parse-depth']
        parsing_results[run]['0']['parse-distance'] = parsing_results_ud['run1']['0']['parse-distance']


    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4, sort_keys=True)


if __name__ == '__main__':
    """
    """

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    parser = ArgumentParser()

    parser.add_argument("--probe_name", default=None, type=str, required=True, help="naacl_19_ptb")
    parser.add_argument("--models_path", default=None, type=str, required=True, help="/home/lpmayos/hd/code/transformers/lpmayos_experiments/bert_base_cased_finetuned_parsing")
    parser.add_argument('--runs', nargs='+', required=True, help="run1 run2 run3 run4 run5")
    parser.add_argument("--output_file", default=None, type=str, required=True, help="bert_base_cased_finetuned_parsing_results.json")

    args = parser.parse_args()
    analyse_results(args.probe_name, args.models_path, args.runs, args.output_file)

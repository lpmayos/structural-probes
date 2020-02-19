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

        run_results_path = models_path + '/' + run + '/results_glue'

        results[run] = {}

        glue_tasks = [os.path.join(run_results_path, o) for o in os.listdir(run_results_path) if os.path.isdir(os.path.join(run_results_path, o))]

        for task in glue_tasks:

            task_name = task.split('/')[-1]

            results[run][task_name] = {}

            task_checkpoints = [os.path.join(task, o) for o in os.listdir(task) if os.path.isdir(os.path.join(task, o)) and o.startswith(('checkpoint-'))]

            for checkpoint_path in task_checkpoints:

                checkpoint = checkpoint_path.split('/')[-1].split('-')[-1]

                logging.info('Reading results for checkpoint %s' % checkpoint)

                results_path = checkpoint_path + '/eval_results.txt'
                probes_path = checkpoint_path + '/structural_probes/' + probe_name

                checkpoint_results = {'task_acc': None,
                                      'task_f1': None,
                                      'task_acc_and_f1': None,
                                      'parse-depth': {
                                          'dev.root_acc': None,
                                          'dev.spearmanr-5_50-mean': None
                                      },
                                      'parse-distance': {
                                          'dev.spearmanr-5_50-mean': None,
                                          'dev.uuas': None
                                      }}

                if os.path.exists(results_path):
                    with open(results_path) as file:
                        for line in file.readlines():
                            parts = line.split(' = ')
                            checkpoint_results[parts[0]] = parts[1]
                else:
                    logging.info('File %s does not exists yet' % results_path)

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

                results[run][task_name][int(checkpoint)] = checkpoint_results

    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4, sort_keys=True)


if __name__ == '__main__':
    """
    """

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    parser = ArgumentParser()

    parser.add_argument("--probe_name", default=None, type=str, required=True, help="naacl_19_ptb")
    parser.add_argument("--models_path", default=None, type=str, required=True, help="/home/lpmayos/hd/code/transformers/lpmayos_experiments/bert_base_cased_finetuned_glue")
    parser.add_argument('--runs', nargs='+', required=True, help="run1 run2 run3 run4 run5")
    parser.add_argument("--output_file", default=None, type=str, required=True, help="bert_base_cased_finetuned_glue_results.json")

    args = parser.parse_args()
    analyse_results(args.probe_name, args.models_path, args.runs, args.output_file)

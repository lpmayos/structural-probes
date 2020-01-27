from __future__ import print_function

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

            try:
                checkpoint_path = run_path + '/results/checkpoint-' + checkpoint
                squad_results_path = checkpoint_path + '/squad_results.json'
                probes_path = checkpoint_path + '/structural_probes/' + probe_name

                checkpoint_results = {'squad_exact_match': None,
                                      'squad_f1': None,
                                      'parse-depth': {},
                                      'parse-distance': {}}

                with open(squad_results_path) as file:
                    data = json.load(file)
                    checkpoint_results['squad_exact_match'] = data['squad_exact_match']
                    checkpoint_results['squad_f1'] = data['squad_f1']

                with open(probes_path + '/parse-depth/dev.root_acc') as file:
                    result = file.readlines()[0].split()[0]  # i.e. 0.8123529411764706      1381    1700
                    checkpoint_results['parse-depth']['dev.root_acc'] = float(result)
                with open(probes_path + '/parse-depth/dev.spearmanr-5_50-mean') as file:
                    result = file.readlines()[0]  # i.e. 0.8552411954331226
                    checkpoint_results['parse-depth']['dev.spearmanr-5_50-mean'] = float(result)

                with open(probes_path + '/parse-distance/dev.uuas') as file:
                    result = file.readlines()[0]  # i.e. 0.7040907201804905
                    checkpoint_results['parse-distance']['dev.uuas'] = result
                with open(probes_path + '/parse-distance/dev.spearmanr-5_50-mean') as float(result):
                    result = file.readlines()[0]  # i.e. 0.8058858201615192
                    checkpoint_results['parse-distance']['dev.spearmanr-5_50-mean'] = float(result)

                results[run]['checkpoint-' + checkpoint] = checkpoint_results

            except FileNotFoundError:
                logging.info('ATTENTION! Could not process results for %s' % checkpoint_path)

    with open(output_file, 'w') as outfile:
        json.dump(results, outfile)


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    parser = ArgumentParser()

    parser.add_argument("--probe_name", default=None, type=str, required=True, help="naacl_19_ptb")
    parser.add_argument("--models_path", default=None, type=str, required=True, help="/home/lpmayos/hd/code/transformers/lpmayos_experiments/bert_base_cased_finetuned_squad")
    parser.add_argument('--runs', nargs='+', required=True, help="run1 run2 run3 run4 run5")
    parser.add_argument('--checkpoints', nargs='+', required=True, help="250 2000 3750 5500 7250 9000 11000 12750 14500 16500 18250 20000 22000")
    parser.add_argument("--output_file", default=None, type=str, required=True, help="bert_base_cased_finetuned_squad_results.json")

    args = parser.parse_args()
    analyse_results(args.probe_name, args.models_path, args.runs, args.checkpoints, args.output_file)

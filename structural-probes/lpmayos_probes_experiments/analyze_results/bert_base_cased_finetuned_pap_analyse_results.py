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

        # load checkpoints results
        run_output_path = models_path + '/' + run + '/output/'

        run_results_path = run_output_path + 'eval_results.json'
        with open(run_results_path) as json_file:
            run_results = json.load(json_file)

        # add checkpoint-0 SRL results

        with open(models_path + '/run0/output/eval_results.json') as json_file:
            base_results = json.load(json_file)
            run_results[0] = base_results['checkpoint_0']

        # add checkpoint-0 probes results
        # We simply copy the results from another task (i.e. parsing)

        with open('bert_base_cased_finetuned_parsing_results.json') as json_file:
            parsing_results = json.load(json_file)

        run_results[0]['parse-depth'] = parsing_results['run1']['0']['parse-depth']
        run_results[0]['parse-distance'] = parsing_results['run1']['0']['parse-distance']

        # replace keys with numbers
        run_results_new = {}
        for checkpoint in run_results:
            checkpoint_name = int(checkpoint.split('/')[1].replace('checkpoint-', ''))
            run_results_new[checkpoint_name] = run_results[checkpoint]
        run_results = run_results_new

        # add probes results for all checkpoints

        for checkpoint in run_results:

            if '/' in checkpoint:
                checkpoint_path = checkpoint.split('/')[1]

                logging.info('Reading results for checkpoint %s' % checkpoint_path)

                probes_path = run_output_path + checkpoint_path + '/structural_probes/' + probe_name

                run_results[checkpoint]['parse-depth'] = {'dev.root_acc': None,
                                                          'dev.spearmanr-5_50-mean': None}
                run_results[checkpoint]['parse-distance'] = {'dev.spearmanr-5_50-mean': None,
                                                             'dev.uuas': None}

                dev_root_acc_file = probes_path + '/parse-depth/dev.root_acc'
                if os.path.exists(dev_root_acc_file):
                    with open(dev_root_acc_file) as file:
                        result = file.readlines()[0].split()[0]  # i.e. 0.8123529411764706      1381    1700
                        run_results[checkpoint]['parse-depth']['dev.root_acc'] = float(result)
                else:
                    logging.info('File %s does not exists yet' % dev_root_acc_file)

                dev_spearmanr_file = probes_path + '/parse-depth/dev.spearmanr-5_50-mean'
                if os.path.exists(dev_spearmanr_file):
                    with open(dev_spearmanr_file) as file:
                        result = file.readlines()[0]  # i.e. 0.8552411954331226
                        run_results[checkpoint]['parse-depth']['dev.spearmanr-5_50-mean'] = float(result)
                else:
                    logging.info('File %s does not exists yet' % dev_spearmanr_file)

                dev_uuas_file = probes_path + '/parse-distance/dev.uuas'
                if os.path.exists(dev_uuas_file):
                    with open(dev_uuas_file) as file:
                        result = file.readlines()[0]  # i.e. 0.7040907201804905
                        run_results[checkpoint]['parse-distance']['dev.uuas'] = float(result)
                else:
                    logging.info('File %s does not exists yet' % dev_uuas_file)

                dev_spearman_file = probes_path + '/parse-distance/dev.spearmanr-5_50-mean'
                if os.path.exists(dev_spearman_file):
                    with open(dev_spearman_file) as file:
                        result = file.readlines()[0]  # i.e. 0.8058858201615192
                        run_results[checkpoint]['parse-distance']['dev.spearmanr-5_50-mean'] = float(result)
                else:
                    logging.info('File %s does not exists yet' % dev_spearman_file)

            results[run] = run_results

    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4, sort_keys=True)


if __name__ == '__main__':
    """
    """

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    parser = ArgumentParser()

    parser.add_argument("--probe_name", default=None, type=str, required=True, help="naacl_19_ptb")
    parser.add_argument("--models_path", default=None, type=str, required=True, help="/homedtic/lperez/parsing-as-pretraining/runs_constituency_parsing")
    parser.add_argument('--runs', nargs='+', required=True, help="run1 run2 run3 run4 run5")
    parser.add_argument("--output_file", default=None, type=str, required=True, help="bert_base_cased_finetuned_pap_constituents_results.json")

    args = parser.parse_args()
    analyse_results(args.probe_name, args.models_path, args.runs, args.output_file)

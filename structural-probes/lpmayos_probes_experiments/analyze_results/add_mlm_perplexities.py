from __future__ import print_function

import os
from argparse import ArgumentParser
import json
import logging
import re


def add_perplexities(models_path, output_file, results_folder):
    """
    """

    with open(output_file, 'r') as f:
        results = json.load(f)

    for i, run in results.items():
        for j, checkpoint in run.items():

            checkpoint_path = models_path + '/' + i + '/' + results_folder + '/checkpoint-' + j
            mlm_results_path = checkpoint_path + '/eval_mlm_results.txt'

            if os.path.exists(mlm_results_path):
                with open(mlm_results_path) as file:
                    line = file.readline()
                    pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+")  # regular expression to match floats, with optional +/-
                    mlm_perplexity = float(pattern.findall(line)[0])
                    checkpoint['mlm_perplexity'] = mlm_perplexity
            else:
                logging.info('File %s does not exists yet' % mlm_results_path)
                checkpoint['mlm_perplexity'] = None

        # transform dictionary keys to integers for propoer sorting when writing to file
        results[i] = {int(k): v for k, v in run.items()}

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    """
    """

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    parser = ArgumentParser()

    parser.add_argument("--models_path", default=None, type=str, required=True, help="/home/lpmayos/hd/code/transformers/lpmayos_experiments/bert_base_cased_finetuned_parsing")
    parser.add_argument("--output_file", default=None, type=str, required=True, help="bert_base_cased_finetuned_parsing_ptb_results.json")
    parser.add_argument("--results_folder", default=None, type=str, required=True, help="reuslts_parsing")

    args = parser.parse_args()
    add_perplexities(args.models_path, args.output_file, args.results_folder)

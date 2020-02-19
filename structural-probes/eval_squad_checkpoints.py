from __future__ import print_function

from argparse import ArgumentParser
import json
import logging
import os

from squad_evaluate_v1_1 import evaluate


def evaluate_squad(dataset_file, prediction_file):

    logging.info('\tEvaluating squad predictions in %s' % prediction_file)

    with open(dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

        with open(prediction_file) as prediction_file:
            predictions = json.load(prediction_file)

            return evaluate(dataset, predictions)


def eval_squad_checkpoints(seed, probe_name, ptb_path, probes_input_paths, parse_distance_yaml, parse_depth_yaml, models_path, checkpoints_list, dataset_file, bert_model, model_to_load):
    """ traverses the given path looking for prediction files and computes the results
    """

    for checkpoint in checkpoints_list:

        predictions_file = models_path + '/results/predictions_' + checkpoint + '.json'
        checkpoint_path = models_path + '/results/checkpoint-' + checkpoint
        squad_results_path = checkpoint_path + '/squad_results.json'

        # 1. evaluate squad

        # evaluate just if results were not already in results.json
        if not os.path.exists(squad_results_path):
            squad_eval = evaluate_squad(dataset_file, predictions_file)
            results = {'squad_exact_match': squad_eval['exact_match'],
                       'squad_f1': squad_eval['f1']}
            with open(squad_results_path, 'w') as f:
                json.dump(results, f, indent=4, sort_keys=True)
        else:
            logging.info('squad results for checkpoint %s already exists; skipping' % checkpoint)


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    parser = ArgumentParser()

    parser.add_argument("--models_path", default=None, type=str, required=True, help="/home/lpmayos/hd/code/transformers/lpmayos_experiments/bert_base_cased_finetuned_squad/run1")
    parser.add_argument('--checkpoints', nargs='+', required=True, help="250 5500 11000 16500 22000")
    parser.add_argument("--squad_dataset_file", default=None, type=str, required=True, help="/home/lpmayos/hd/code/transformers/examples/tests_samples/SQUAD/dev-v1.1.json")

    args = parser.parse_args()
    eval_squad_checkpoints(args.models_path, args.checkpoints, args.squad_dataset_file)

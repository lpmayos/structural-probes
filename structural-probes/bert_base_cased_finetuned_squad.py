from __future__ import print_function

from argparse import ArgumentParser
import json
import logging
import os

import h5py
import torch
import numpy as np

import yaml

from transformers import (
    BertForQuestionAnswering, # TODO lpmayos I use normal BertModel to generate hdf5 files; maybe for other use cases I need this other model (seems to work loading one into the other)
    BertModel,
    BertTokenizer,
)

from run_experiment import setup_new_experiment_dir, execute_experiment
from squad_evaluate_v1_1 import evaluate


def evaluate_squad(dataset_file, prediction_file):

    logging.info('\tEvaluating squad predictions in %s' % prediction_file)

    with open(dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

        with open(prediction_file) as prediction_file:
            predictions = json.load(prediction_file)

            return evaluate(dataset, predictions)


def convert_raw_to_bert_hdf5(model_path, probes_input_paths, output_folder_path, bert_model, model_to_load):
    """ Copied from scripts/convert_raw_to_bert.py
    """
    hdf5_files_paths = []

    if not model_to_load:
        model = BertModel.from_pretrained(model_path)
    else:
        logging.info("ATTENTION!! Provided model name to load: %s" % model_to_load)
        model = BertModel.from_pretrained(model_to_load)

    # Load pre-trained model tokenizer (vocabulary)
    # Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
    if bert_model == 'base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        LAYER_COUNT = 12
        FEATURE_COUNT = 768
    elif bert_model == 'large':
        tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        LAYER_COUNT = 24
        FEATURE_COUNT = 1024
    else:
        raise ValueError("BERT model must be base or large")

    model.eval()

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for input_path in probes_input_paths:

        output_path = output_folder_path + '/' + input_path.split('/')[-1].replace('.txt', '.hdf5')
        hdf5_files_paths.append(output_path)
        if not os.path.exists(output_path):
            with h5py.File(output_path, 'w') as fout:

                logging.info('Generating hdf5 file in %s.' % (output_path))

                for index, line in enumerate(open(input_path)):
                    line = line.strip()  # Remove trailing characters
                    line = '[CLS] ' + line + ' [SEP]'
                    tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(line)
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                    segment_ids = [1 for x in tokenized_text]

                    # Convert inputs to PyTorch tensors
                    tokens_tensor = torch.tensor([indexed_tokens])
                    segments_tensors = torch.tensor([segment_ids])

                    with torch.no_grad():
                        encoded_layers, _ = model(tokens_tensor, segments_tensors)
                    dset = fout.create_dataset(str(index), (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT))
                    dset[:, :, :] = np.vstack([np.array(x) for x in encoded_layers])
        else:
            logging.info('ATTENTION! hdf5 file %s already existed! skipping hdf5 generation.' % (output_path))

    return hdf5_files_paths


def remove_generated_hdf5(hdf5_files_paths):
    for file in hdf5_files_paths:
        if os.path.exists(file):
            os.remove(file)
            logging.info("Removed file in %s" % file)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def execute_probe(seed, configuration, probes_path_hdf5, ptb_path, results_dir, config_file):
    if not os.path.exists(results_dir):
        configuration['dataset']['corpus']['root'] = ptb_path
        configuration['dataset']['embeddings']['root'] = probes_path_hdf5
        configuration['dataset']['embeddings']['train_path'] = 'train.gold.hdf5'
        configuration['dataset']['embeddings']['dev_path'] = 'dev.gold.hdf5'
        configuration['dataset']['embeddings']['test_path'] = 'test.gold.hdf5'
        configuration['dataset']['corpus']['train_path'] = 'train.gold.conll'
        configuration['dataset']['corpus']['dev_path'] = 'dev.gold.conll'
        configuration['dataset']['corpus']['test_path'] = 'test.gold.conll'
        configuration['reporting']['root'] = results_dir

        if not os.path.exists(config_file):
            with open(config_file, 'w', encoding='utf8') as outfile:
                yaml.dump(configuration, outfile, default_flow_style=False, allow_unicode=True)
        else:
            logging.info(config_file + ' already exists; skipping creation')

        cli_args = Namespace(experiment_config=config_file,
                             results_dir=results_dir,
                             train_probe=1,
                             report_results=1,
                             seed=seed)

        # set all random seeds for (within-machine) reproducibility
        logging.info('Setting random seed to %s for (within-machine) reproducibility' % seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        yaml_args = yaml.load(open(config_file))
        setup_new_experiment_dir(cli_args, yaml_args, cli_args.results_dir)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        yaml_args['device'] = device
        execute_experiment(yaml_args, train_probe=cli_args.train_probe, report_results=cli_args.report_results)
    else:
        logging.info('ATTENTION: syntactic probes results folder %s already exists; skipping probing' % results_dir)


def eval_squad_and_structural_probing(seed, probe_name, ptb_path, probes_input_paths, parse_distance_yaml, parse_depth_yaml, models_path, checkpoints_list, dataset_file, bert_model, model_to_load):
    """ traverses the given path looking for prediction files and computes the results
    """

    # load example yaml config files for later
    with open(parse_distance_yaml, 'r') as pad_yaml:
        pad_yaml = yaml.safe_load(pad_yaml)
    with open(parse_depth_yaml, 'r') as prd_yaml:
        prd_yaml = yaml.safe_load(prd_yaml)

    for checkpoint in checkpoints_list:

        predictions_file = models_path + '/results/predictions_' + checkpoint + '.json'
        checkpoint_path = models_path + '/results/checkpoint-' + checkpoint
        squad_results_path = checkpoint_path + '/squad_results.json'
        probes_path = checkpoint_path + '/structural_probes/' + probe_name
        probes_path_hdf5 = checkpoint_path + '/structural_probes/' + probe_name + '/hdf5'
        if not os.path.exists(probes_path_hdf5):
            os.makedirs(probes_path_hdf5)

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

        # 2. run probes

        parse_depth_results_dir = probes_path + '/parse-depth'
        parse_distance_results_dir = probes_path + '/parse-distance'

        if not os.path.exists(parse_depth_results_dir) or not os.path.exists(parse_distance_results_dir):

            # 2.1. Generate hdf5 file with model

            hdf5_files_paths = convert_raw_to_bert_hdf5(checkpoint_path, probes_input_paths, probes_path_hdf5, bert_model, model_to_load)

            # 2.2. Execute probes using the generated hdf5 files (copied from structural-probes/run_experiment.py)

            config_file = probes_path + '/parse_depth.yaml'
            execute_probe(seed, pad_yaml, probes_path_hdf5, ptb_path, parse_depth_results_dir, config_file)

            config_file = probes_path + '/parse_distance.yaml'
            execute_probe(seed, prd_yaml, probes_path_hdf5, ptb_path, parse_distance_results_dir, config_file)

            # 2.3. Remove generated hdf5 files

            remove_generated_hdf5(hdf5_files_paths)

        else:
            logging.info('ATTENTION: syntactic probes results folders already exists; skipping probing')


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    parser = ArgumentParser()

    parser.add_argument("--probe_name", default=None, type=str, required=True, help="naacl_19_ptb")
    parser.add_argument("--ptb_path", default=None, type=str, required=True, help="/home/lpmayos/hd/code/datasets/PTB_SD_3_3_0/")
    parser.add_argument('--probes_input_paths', nargs='+', required=True, help="/home/lpmayos/hd/code/datasets/PTB_SD_3_3_0_mini/train.gold.txt /home/lpmayos/hd/code/datasets/PTB_SD_3_3_0_mini/dev.gold.txt /home/lpmayos/hd/code/datasets/PTB_SD_3_3_0_mini/test.gold.txt")
    parser.add_argument("--parse_distance_yaml", default=None, type=str, required=True, help="/home/lpmayos/hd/code/structural-probes/example/config/naacl19/bert-base/ptb-pad-BERTbase7.yaml")
    parser.add_argument("--parse_depth_yaml", default=None, type=str, required=True, help="/home/lpmayos/hd/code/structural-probes/example/config/naacl19/bert-base/ptb-prd-BERTbase7.yaml")
    parser.add_argument("--models_path", default=None, type=str, required=True, help="/home/lpmayos/hd/code/transformers/lpmayos_experiments/bert_base_cased_finetuned_squad/run1")
    parser.add_argument('--checkpoints', nargs='+', required=True, help="250 5500 11000 16500 22000")
    parser.add_argument("--squad_dataset_file", default=None, type=str, required=True, help="/home/lpmayos/hd/code/transformers/examples/tests_samples/SQUAD/dev-v1.1.json")
    parser.add_argument("--bert_type", default="base", type=str, required=False, help="base or large")
    parser.add_argument("--model_to_load", default=None, type=str, required=False, help="If provided, we load this model instead of the models in the checkpoints")
    parser.add_argument("--seed", type=int, required=True, help="sets all random seeds for (within-machine) reproducibility")

    args = parser.parse_args()
    eval_squad_and_structural_probing(args.seed, args.probe_name, args.ptb_path, args.probes_input_paths, args.parse_distance_yaml, args.parse_depth_yaml, args.models_path, args.checkpoints, args.squad_dataset_file, args.bert_type, args.model_to_load)

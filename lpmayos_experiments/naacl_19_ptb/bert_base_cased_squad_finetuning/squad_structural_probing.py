from __future__ import print_function
from collections import Counter
import string
import re
import json
import sys
import logging
import os

import h5py
import torch
import numpy as np

import yaml

from transformers import (
    BertConfig,
    BertForQuestionAnswering, # TODO lpmayos I use normal BertModel to generate hdf5 files; maybe for other use cases I need this other model (seems to work loading one into the other)
    BertModel,
    BertTokenizer,
)


# -----------------------------------------------------------Official evaluation script for v1.1 of the SQuAD dataset.
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}
# ----------------------------------------------------------- end Official evaluation script for v1.1 of the SQuAD dataset.


def evaluate_squad(dataset_file, prediction_file):

    logging.info('\tEvaluating squad predictions in %s' % prediction_file)

    with open(dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

        with open(prediction_file) as prediction_file:
            predictions = json.load(prediction_file)

            return evaluate(dataset, predictions)


def convert_raw_to_bert_hdf5(model_path, probes_input_paths, output_folder_path, bert_model='base', do_lower_case=False):
    """ Copied from scripts/convert_raw_to_bert.py
    """
    model = BertModel.from_pretrained(model_path)

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


def eval_squad_and_structural_proing(ptb_path, probes_input_paths, parse_distance_yaml, parse_depth_yaml, models_path, checkpoints_list, dataset_file):
    """ traverses the given path looking for prediction files and computes the results
    """

    # load yaml config files for the probes

    with open(parse_distance_yaml, 'r') as pad_yaml:
        pad_yaml = yaml.safe_load(pad_yaml)
    with open(parse_depth_yaml, 'r') as prd_yaml:
        prd_yaml = yaml.safe_load(prd_yaml)

    # prepare results folders and files

    results_folder = models_path + '/structural_probes'
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    results_path = results_folder + '/results.json'
    if not os.path.exists(results_path):
        results = {'models_path': models_path}
    else:
        with open(results_path) as f:
            logging.info('%s already exists; loading' % results_path)
            results = json.load(f)

    for checkpoint in checkpoints_list:

        if checkpoint not in results:
            results[checkpoint] = {}

        predictions_file = models_path + '/results/predictions_' + checkpoint + '.json'
        checkpoint_path = models_path + '/results/checkpoint-' + checkpoint

        probes_path = checkpoint_path + '/hdf5_naacl_19_ptb'
        if not os.path.exists(probes_path):
            os.mkdir(probes_path)

        # 1. evaluate squad

        # evaluate just if results were not already in results.json
        if 'squad_exact_match' not in results[checkpoint]:
            squad_eval = evaluate_squad(dataset_file, predictions_file)
            results[checkpoint] = {'squad_exact_match': squad_eval['exact_match'],
                                   'squad_f1': squad_eval['f1']}
        else:
            logging.info('squad results for checkpoint %s already exists; skipping' % checkpoint)

        # 2. run probes

        # 2.1. Generate hdf5 file with model

        # convert_raw_to_bert_hdf5(model_path, probes_input_paths, probes_path)  # TODO uncomment

        # 2.2. Execute probes using the generated hdf5 files

        if not os.path.exists(probes_path + '/parse_depth.yaml'):
            pad_yaml['dataset']['corpus']['root'] = ptb_path
            pad_yaml['dataset']['embeddings']['root'] = probes_path
            pad_yaml['reporting']['root'] = probes_path + '/probing/parse-depth'
            with open(probes_path + '/parse_depth.yaml', 'w', encoding='utf8') as outfile:
                yaml.dump(pad_yaml, outfile, default_flow_style=False, allow_unicode=True)
        else:
            logging.info(probes_path + '/parse_depth.yaml already exists; skipping creation')

        if not os.path.exists(probes_path + '/parse_distance.yaml'):
            prd_yaml['dataset']['corpus']['root'] = ptb_path
            prd_yaml['dataset']['embeddings']['root'] = probes_path
            prd_yaml['reporting']['root'] = probes_path + '/probing/parse-distance'
            with open(probes_path + '/parse_distance.yaml', 'w', encoding='utf8') as outfile:
                yaml.dump(prd_yaml, outfile, default_flow_style=False, allow_unicode=True)
        else:
            logging.info(probes_path + '/parse_distance.yaml already exists; skipping creation')

        # TODO execute run_experiment:
        #   /home/lpmayos/hd/code/structural-probes/structural-probes/run_experiment.py path_to_yaml.yaml

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(message)s")

    ptb_path = '/home/lpmayos/hd/code/datasets/PTB_SD_3_3_0/'
    probes_input_paths = ['/home/lpmayos/hd/code/datasets/PTB_SD_3_3_0/train.gold.txt',
                          '/home/lpmayos/hd/code/datasets/PTB_SD_3_3_0/dev.gold.txt',
                          '/home/lpmayos/hd/code/datasets/PTB_SD_3_3_0/test.gold.txt']

    parse_distance_yaml = '/home/lpmayos/hd/code/structural-probes/lpmayos_tests_K&G/config/naacl2019/ptb-pad-BERTbase7.yaml'
    parse_depth_yaml = '/home/lpmayos/hd/code/structural-probes/lpmayos_tests_K&G/config/naacl2019/ptb-prd-BERTbase7.yaml'

    models_path = '/home/lpmayos/hd/code/transformers/lpmayos_experiments/bert_base_cased_finetuned_squad/run1'
    checkpoints = ['250', '5500', '11000', '16500', '22000']  # TODO restricted to 5 probings per model, to save time and space
    # checkpoints = ['250']  # TODO restrict back to 5 probings per model

    squad_dataset_file = '/home/lpmayos/hd/code/transformers/examples/tests_samples/SQUAD/dev-v1.1.json'

    eval_squad_and_structural_proing(ptb_path, probes_input_paths, parse_distance_yaml, parse_depth_yaml, models_path, checkpoints, squad_dataset_file)

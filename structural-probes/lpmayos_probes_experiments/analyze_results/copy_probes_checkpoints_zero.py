from __future__ import print_function

import os
import shutil
from argparse import ArgumentParser
import logging


def copy_probes_checkpoints_zero(models_path, probes_checkpoint_zero_path):
    probes_folder = probes_checkpoint_zero_path.split('/')[-1]
    for root, dirs, files in os.walk(models_path, topdown=False):
        for dir_name in dirs:
            if dir_name.startswith('checkpoint-0'):
                dir_path = '/'.join([root, dir_name, probes_folder])
                logging.debug('copying %s into %s' % (probes_checkpoint_zero_path, dir_path))
                try:
                    shutil.copytree(probes_checkpoint_zero_path, dir_path)
                except FileExistsError:
                    logging.info('File already exists!')


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    parser = ArgumentParser()

    parser.add_argument("--models_path", default=None, type=str, required=True,
                        help="/homedtic/lperez/transformers/lpmayos_experiments")
    parser.add_argument("--probes_checkpoint_zero_path", default=None, type=str, required=True,
                        help="/homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_squad/run0/results/checkpoint-0/structural_probes")

    args = parser.parse_args()
    copy_probes_checkpoints_zero(args.models_path, args.probes_checkpoint_zero_path)

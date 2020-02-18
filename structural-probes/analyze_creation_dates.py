import os
import logging
from argparse import ArgumentParser
from datetime import datetime
import re


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG,
                        format="%(message)s")

    argp = ArgumentParser()
    argp.add_argument('root_dir', type=str, help='/homedtic/lperez/transformers/lpmayos_experiments/bert_base_cased_finetuned_squad/run1')
    argp.add_argument('file_name', type=str, help='squad_results.json')
    cli_args = argp.parse_args()

    pattern = re.compile(r"-[0-9]+\/")

    results = {}

    for dirName, subdirList, fileList in os.walk(cli_args.root_dir):
        for fname in fileList:
            if fname == cli_args.file_name:
                fpath = '/'.join([dirName, fname])
                checkpoint = float(pattern.findall(fpath)[0].replace('-', '').replace('/', ''))
                time_str = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%d/%m/%Y %H:%M:%S")
                results[checkpoint] = {'timestamp': time_str, 'path': fpath}
 
    for file_found in sorted(results.keys()):
        logging.info('%s modified on \t%s' % (results[file_found]['path'], results[file_found]['timestamp']))



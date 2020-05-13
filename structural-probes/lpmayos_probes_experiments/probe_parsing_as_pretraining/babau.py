import os


checkpoints_path = '/homedtic/lperez/parsing-as-pretraining/runs_constituency_parsing/run1/output'
model_partial_name = 'pytorch_model'

for (root, dirs, files) in os.walk(checkpoints_path):
    for file in files:
        if file.startswith(model_partial_name):
            model_path = root + '/' + file
            print('Evaluating %s' % model_path)
            checkpoint_path = '/'.join(model_path.split('/')[0:-1])
            print('checkpoint_path = %s' % checkpoint_path)


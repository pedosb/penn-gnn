import argparse
from glob import glob

import numpy as np
import torch

import lab3_models

parser = argparse.ArgumentParser()
parser.add_argument('--experiments', nargs='+', help='Which experiments to run')
parser.add_argument('--dataset-glob-pattern',
                    default=r"datasets/lab3_similarity_??.pt",
                    help='Run training on each of the matched datasets report the average result')
parser.add_argument('--show-model-summary', action='store_true')
args = parser.parse_args()

experiments = {
    'Linear': {
        'task': 'linear'
    },
    'FCNN': {
        'task': 'fcnn'
    },
    'Filter': {
        'task': 'graph_filter'
    },
    'GNN': {
        'task': 'gnn'
    },
    '2-layers GNN': {
        'task': '2l_gnn'
    }
}

n_experiments = 1
for name, experiment_args in experiments.items():
    if args.experiments is not None and name not in args.experiments:
        continue

    results = []
    for _ in range(n_experiments):
        for dataset_name in glob(args.dataset_glob_pattern):
            dataset_args = torch.load(dataset_name)

            model_name = experiment_args['task']
            experiment_args['save_prefix'] = f'{model_name}_{len(results):02d}'
            experiment_args['show_summary'] = args.show_model_summary and len(results) == 0
            results.append(lab3_models.run_experiment(
                *dataset_args,
                **experiment_args,
            ))
    results = np.array(results)
    print('Train')
    print(f'{np.mean(results[:, 1]):.5f} - {name} -',
          ''.join([f'{v:.3f} ' for v in results[:, 1]]))
    print('Test')
    print(f'{np.mean(results[:, 0]):.5f} - {name} -',
          ''.join([f'{v:.3f} ' for v in results[:, 0]]))

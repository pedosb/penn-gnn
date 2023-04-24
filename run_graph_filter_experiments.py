import argparse
from glob import glob

import numpy as np
import torch
from mainOnlyFilter import mainOnlyFilter
from mainOnlyPerceptron import mainOnlyPerceptron

import lab2_graph_filter_train

parser = argparse.ArgumentParser()
parser.add_argument('--experiments', nargs='+', help='Which experiments to run')
args = parser.parse_args()

experiments = {
    'Reference filter': mainOnlyFilter.run,
    'Reference perceptron': mainOnlyPerceptron.run,
    'My filter': {
        'task': 'graph_filter'
    },
    'My perceptron': {
        'task': 'graph_perceptron'
    },
    'My MLGNN': {
        'task': 'multilayer_gnn'
    },
    'My multi-feature graph filter': {
        'task': 'multi_feature_graph_filter'
    },
    'My multi-feature 2-layers GNN': {
        'task': 'multi_feature_two_layers_gnn'
    },
    'My multi-feature 3-layers GNN': {
        'task': 'multi_feature_three_layers_gnn'
    },
    'My linear': {
        'task': 'linear'
    },
}

n_experiments = 1
for name, experiment_args in experiments.items():
    if args.experiments is not None and name not in args.experiments:
        continue

    result_sum = 0
    total_results = 0
    for _ in range(n_experiments):
        for dataset_name in glob(r"lab2_dataset_*.pt"):
            dataset_args = torch.load(dataset_name)

            experiment_args['save_fig'] = total_results == 0
            experiment_args['show_summary'] = total_results == 0
            result_sum += lab2_graph_filter_train.run(
                *dataset_args,
                **experiment_args,
            )
            total_results += 1
    result_avg = result_sum / total_results
    print(f'{result_avg} - {name}')

import argparse
import os
import re
from glob import glob

import numpy as np
import torch

import lab3_models
from training import evaluate_model_loss

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

    results = {}
    results['150_reviews'] = []
    results['50_reviews'] = []
    results['10_reviews'] = []
    transferability_results = ['50_reviews', '10_reviews']
    for _ in range(n_experiments):
        for i_dataset, dataset_name in enumerate(glob(args.dataset_glob_pattern)):
            dataset_args = torch.load(dataset_name)

            model_name = experiment_args['task']
            experiment_args['save_prefix'] = f'{model_name}_{i_dataset:02d}'
            experiment_args['show_summary'] = args.show_model_summary and i_dataset == 0
            results['150_reviews'].append(
                lab3_models.run_experiment(
                    *dataset_args,
                    **experiment_args,
                ))

            for n_reviews_str in transferability_results:
                model = torch.load(os.path.join('models', experiment_args['save_prefix'] + '.pt'))
                target_movie, adjacency_matrix, _, X_test, _, _, Y_test, _ = torch.load(
                    re.sub(r'150_reviews(\.pt$)', rf'{n_reviews_str}\1', dataset_name))
                model.set_graph_shift_operator(adjacency_matrix)
                test_loss = evaluate_model_loss(model, lab3_models.make_eval_loss(target_movie),
                                                X_test, Y_test)[0]
                results[n_reviews_str].append(test_loss)

    result_array = np.array(results['150_reviews'])
    print('Train')

    def show_result(name, result):
        print(f'{np.mean(result):.5f} - {name} -', ''.join([f'{v:.3f} ' for v in result]))

    show_result(name, result_array[:, 1])
    print('Test')
    show_result(name, result_array[:, 0])

    for n_reviews_str in transferability_results:
        result_array = np.array(results[n_reviews_str])
        show_result(n_reviews_str, result_array)

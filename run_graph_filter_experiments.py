from mainOnlyFilter import mainOnlyFilter
from mainOnlyPerceptron import mainOnlyPerceptron
import lab2_graph_filter_train

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiments', nargs='+', help='Which experiments to run')
args = parser.parse_args()

experiments = {
    'Reference filter':
    mainOnlyFilter.run,
    'Reference perceptron':
    mainOnlyPerceptron.run,
    'My filter':
    lambda: lab2_graph_filter_train.run('graph_filter'),
    'My perceptron':
    lambda: lab2_graph_filter_train.run('graph_perceptron'),
    'My MLGNN':
    lambda: lab2_graph_filter_train.run('multilayer_gnn'),
    'My multi-feature graph filter':
    lambda: lab2_graph_filter_train.run('multi_feature_graph_filter'),
    'My multi-feature 2-layers GNN':
    lambda: lab2_graph_filter_train.run('multi_feature_two_layers_gnn'),
    'My multi-feature 3-layers GNN':
    lambda: lab2_graph_filter_train.run('multi_feature_three_layers_gnn', save_fig=True),
}

n_experiments = 30
for name, experiment in experiments.items():
    if args.experiments is not None and name not in args.experiments:
        continue
    result_sum = 0
    for _ in range(n_experiments):
        result_sum += experiment()
    result_avg = result_sum / n_experiments
    print(f'{result_avg} - {name}')

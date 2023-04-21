from mainOnlyFilter import mainOnlyFilter
from mainOnlyPerceptron import mainOnlyPerceptron
import lab2_graph_filter_train

experiments = {
    'Reference filter': mainOnlyFilter.run,
    'Reference perceptron': mainOnlyPerceptron.run,
    'My filter': lambda: lab2_graph_filter_train.run('graph_filter'),
    'My perceptron': lambda: lab2_graph_filter_train.run('graph_perceptron'),
    'My MLGNN': lambda: lab2_graph_filter_train.run('multilayer_gnn'),
}

n_experiments = 30
for name, experiment in experiments.items():
    result_sum = 0
    for _ in range(n_experiments):
        result_sum += experiment()
    result_avg = result_sum / n_experiments
    print(f'{result_avg} - {name}')

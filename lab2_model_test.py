# %%
from glob import glob

import torch
from lab2_graph_filter import MultiLayerGNN

torch.random.manual_seed(20)


def model_test(graph_shift_operator, X_test, Y_test):
    loss_function = torch.nn.MSELoss(reduction='mean')

    for model_file_name in glob('models/*'):
        model = torch.load(model_file_name)  # type: MultiLayerGNN

        model.set_graph_shift_operator(graph_shift_operator)
        model.eval()
        with torch.no_grad():
            predicted = model.forward(X_test).squeeze()
            test_loss = loss_function(Y_test, predicted)
            print(f'{model_file_name} - Test loss: {test_loss}')


if __name__ == '__main__':
    datasets_names = [
        'lab2_dataset_00.pt', 'lab2_dataset_500_nodes.pt', 'lab2_dataset_1000_nodes.pt'
    ]
    for dataset_name in datasets_names:
        print(f'Dataset: {dataset_name}')
        adjacency_matrix, X_train, X_test, X_validation, Y_train, Y_test, Y_validation = torch.load(
            dataset_name)
        model_test(adjacency_matrix, X_test, Y_test)

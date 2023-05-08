# Comments/results of Lab 2

One observation is that for GNN, I had to use leaky RELU, the RELU was getting "stuck" with zero gradients for multiple
training instances. When implementing lab 1 I saw a difference. It seems that this might be caused by using RELU at the output layer (the last one should be linear, this might be what mess the gradients).

Those results are also very unstable and dependent on random numbers, the order which they are executed seems to matter
a lot.

## 5.1 Linear model vs graph filter

For a similar performance, the graph filter uses significantly less parameters.

```txt
$ git checkout lab2-up-to-5.3
$ PYTHONPATH=.:reference_material python run_graph_filter_experiments.py --experiments 'My linear' 'My multi-feature graph filter'
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MultiLayerGNN                            --
├─Sequential: 1-1                        --
│    └─GraphFilter: 2-1                  256
│    └─GraphFilter: 2-2                  32
=================================================================
Total params: 288
Trainable params: 288
Non-trainable params: 0
=================================================================
5.081482123085044 - My multi-feature graph filter
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
Linear                                   2,550
=================================================================
Total params: 2,550
Trainable params: 2,550
Non-trainable params: 0
=================================================================
5.1934262307895676 - My linear
```

## 5.2 FCNN vs GNN

Similar result here. Interesting that both of them are better than the linear versions.

```txt
$ git checkout lab2-up-to-5.3
$ PYTHONPATH=.:reference_material python run_graph_filter_experiments.py --experiments 'My FCNN' 'My multi-feature 2-layers GNN'
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MultiLayerGNN                            --
├─Sequential: 1-1                        --
│    └─GraphFilter: 2-1                  256
│    │    └─LeakyReLU: 3-1               --
│    └─GraphFilter: 2-2                  32
│    │    └─LeakyReLU: 3-2               --
=================================================================
Total params: 288
Trainable params: 288
Non-trainable params: 0
=================================================================
4.969721798023696 - My multi-feature 2-layers GNN
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
Sequential                               --
├─Linear: 1-1                            1,275
├─LeakyReLU: 1-2                         --
├─Linear: 1-3                            1,300
=================================================================
Total params: 2,575
Trainable params: 2,575
Non-trainable params: 0
=================================================================
5.141648419737256 - My FCNN
```

## 5.3 Graph filters vs GNN

The parameters from the GNN to the graph filter are the same. However, GNN perform better. The non-linearity helped in achieving better results. More GNN layers also helped the model.

```txt
$ git checkout lab2-up-to-5.3
$ PYTHONPATH=.:reference_material python run_graph_filter_experiments.py --experiments 'My multi-feature graph filter' 'My multi-feature 2-layers GNN' 'My multi-feature 3-layers GNN'
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MultiLayerGNN                            --
├─Sequential: 1-1                        --
│    └─GraphFilter: 2-1                  256
│    └─GraphFilter: 2-2                  32
=================================================================
Total params: 288
Trainable params: 288
Non-trainable params: 0
=================================================================
5.081482123085044 - My multi-feature graph filter
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MultiLayerGNN                            --
├─Sequential: 1-1                        --
│    └─GraphFilter: 2-1                  256
│    │    └─LeakyReLU: 3-1               --
│    └─GraphFilter: 2-2                  32
│    │    └─LeakyReLU: 3-2               --
=================================================================
Total params: 288
Trainable params: 288
Non-trainable params: 0
=================================================================
4.917052338781955 - My multi-feature 2-layers GNN
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MultiLayerGNN                            --
├─Sequential: 1-1                        --
│    └─GraphFilter: 2-1                  80
│    │    └─LeakyReLU: 3-1               --
│    └─GraphFilter: 2-2                  320
│    │    └─LeakyReLU: 3-2               --
│    └─GraphFilter: 2-3                  4
│    │    └─LeakyReLU: 3-3               --
=================================================================
Total params: 404
Trainable params: 404
Non-trainable params: 0
=================================================================
4.738057688867325 - My multi-feature 3-layers GNN
```

## 5.5 Training on small graphs

```text
$ git checkout lab2-up-to-5.7
$ PYTHONPATH=.:reference_material python run_graph_filter_experiments.py --experiments 'My multi-feature 3-layers GNN' 'My multi-feature 2-layers GNN' --dataset-gl
ob-pattern lab2_dataset_00.pt
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MultiLayerGNN                            --
├─Sequential: 1-1                        --
│    └─GraphFilter: 2-1                  256
│    │    └─LeakyReLU: 3-1               --
│    └─GraphFilter: 2-2                  32
│    │    └─LeakyReLU: 3-2               --
=================================================================
Total params: 288
Trainable params: 288
Non-trainable params: 0
=================================================================
4.601770516103738 - My multi-feature 2-layers GNN
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MultiLayerGNN                            --
├─Sequential: 1-1                        --
│    └─GraphFilter: 2-1                  80
│    │    └─LeakyReLU: 3-1               --
│    └─GraphFilter: 2-2                  320
│    │    └─LeakyReLU: 3-2               --
│    └─GraphFilter: 2-3                  4
│    │    └─LeakyReLU: 3-3               --
=================================================================
Total params: 404
Trainable params: 404
Non-trainable params: 0
=================================================================
4.942086379707882 - My multi-feature 3-layers GNN
```

## 5.6/5.7 transferability to bigger graphs

The loss of the two layers model increased by 20% and 41% for the 500 and 1000 nodes datasets, respectively. While the
three layers increased by 7% and 24%. It could be said that the three layers generalized better, however, it was not as
good from the start. Furthermore, the results do not seems to be very stable. Maybe it could be said that the better
model is performing worst on the other datasets because it is more specialized (changing the random seed can make the
three layers better at first and invert the roles).

```text
$ git checkout lab2-up-to-5.7
$ python lab2_model_test.py
Dataset: lab2_dataset_00.pt
models/multi_feature_three_layers_gnn.pt - Test loss: 4.942086379707882
models/multi_feature_two_layers_gnn.pt - Test loss: 4.601770516103738
Dataset: lab2_dataset_500_nodes.pt
models/multi_feature_three_layers_gnn.pt - Test loss: 5.3235010642264395
models/multi_feature_two_layers_gnn.pt - Test loss: 5.632489237470202
Dataset: lab2_dataset_1000_nodes.pt
models/multi_feature_three_layers_gnn.pt - Test loss: 6.175087830577612
models/multi_feature_two_layers_gnn.pt - Test loss: 6.504705666305659
```

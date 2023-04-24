# Comments/results of Lab 2

One observation is that for GNN, I had to use leaky RELU, the RELU was getting "stuck" with zero gradients for multiple
training instances.

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

The parameters from the GNN to the graph filter are the same. The non-linearity helped in achieving better result.

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
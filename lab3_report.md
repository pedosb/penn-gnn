# Comments/results of Lab 3

It seems that the batch size cannot be too big for GNN, it does not converge. At the same time, the linear model
suffers with lower batch sizes.

## 2.2 Linear model vs graph filter

Interesting that the graph filter now is performing better than the linear model, even with the linear model having way more parameters. It seems that the linear model is overfitting the data. Remember that we are training for more epochs now, when compared to lab2.

```txt
$ git checkout lab3-up-to-2.4
$ python lab3_main.py --experiments "Linear" "Filter" --dataset-glob-pattern "datasets/lab3_similarity_??.pt" --show-model-summary
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
Linear                                   41,412
=================================================================
Total params: 41,412
Trainable params: 41,412
Non-trainable params: 0
=================================================================
Train
0.95525 - Linear - 0.991 0.879 0.868 0.908 0.961 0.892 0.912 1.230
Test
1.74159 - Linear - 1.620 1.638 1.571 1.856 1.929 1.789 2.023 1.507
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MultiLayerGNN                            --
├─Sequential: 1-1                        --
│    └─GraphFilter: 2-1                  325
├─Linear: 1-2                            65
=================================================================
Total params: 390
Trainable params: 390
Non-trainable params: 0
=================================================================
Train
1.00621 - Filter - 1.045 1.063 0.995 0.991 0.978 0.995 0.986 0.996
Test
1.02469 - Filter - 0.866 1.144 1.051 1.133 1.085 0.836 1.177 0.906
```

## 2.3 FCNN vs GNN

The FCNN is better than the linear model and half of its size. The GNN is even smaller, the non-linearity helped.

```txt
$ git checkout lab3-up-to-2.4
$ python lab3_main.py --experiments "FCNN" "GNN" --dataset-glob-pattern "datasets/lab3_similarity_??.pt" --show-model-summary
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
Sequential                               --
├─Linear: 1-1                            13,056
├─ReLU: 1-2                              --
├─Linear: 1-3                            13,195
=================================================================
Total params: 26,251
Trainable params: 26,251
Non-trainable params: 0
=================================================================
Train
0.94986 - FCNN - 0.954 0.955 1.011 0.932 0.913 0.972 0.875 0.988
Test
1.04232 - FCNN - 0.919 0.965 1.061 1.098 1.098 0.822 1.433 0.943
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MultiLayerGNN                            --
├─Sequential: 1-1                        --
│    └─GraphFilter: 2-1                  325
│    │    └─ReLU: 3-1                    --
├─Linear: 1-2                            65
=================================================================
Total params: 390
Trainable params: 390
Non-trainable params: 0
=================================================================
Train
0.99137 - GNN - 1.020 0.994 0.970 0.979 0.987 1.010 0.970 1.001
Test
0.99786 - GNN - 0.860 0.968 1.073 1.055 1.100 0.854 1.171 0.902
```

## 2.4 Graph filter vs GNN

The GNNs are better than the filters. But the two-layers is not better than the one. Is it overfitting?

```txt
$ git checkout lab3-up-to-2.4
$ python lab3_main.py --experiments "Filter" "GNN" "2-layers GNN" --dataset-glob-pattern "datasets/lab3_similarity_??.pt" --show-model-summary
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MultiLayerGNN                            --
├─Sequential: 1-1                        --
│    └─GraphFilter: 2-1                  325
├─Linear: 1-2                            65
=================================================================
Total params: 390
Trainable params: 390
Non-trainable params: 0
=================================================================
Train
1.07856 - Filter - 1.009 1.016 1.024 0.975 1.643 1.002 0.963 0.996
Test
1.06992 - Filter - 0.837 0.926 1.131 1.060 1.690 0.840 1.169 0.907
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MultiLayerGNN                            --
├─Sequential: 1-1                        --
│    └─GraphFilter: 2-1                  325
│    │    └─ReLU: 3-1                    --
├─Linear: 1-2                            65
=================================================================
Total params: 390
Trainable params: 390
Non-trainable params: 0
=================================================================
Train
0.99549 - GNN - 1.032 0.997 0.967 0.983 0.985 1.024 0.976 1.001
Test
0.99634 - GNN - 0.881 0.962 1.065 1.051 1.100 0.832 1.176 0.903
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MultiLayerGNN                            --
├─Sequential: 1-1                        --
│    └─GraphFilter: 2-1                  325
│    │    └─ReLU: 3-1                    --
│    └─GraphFilter: 2-2                  10,245
│    │    └─ReLU: 3-2                    --
├─Linear: 1-2                            33
=================================================================
Total params: 10,603
Trainable params: 10,603
Non-trainable params: 0
=================================================================
Train
0.96672 - 2-layers GNN - 0.978 0.965 0.948 0.981 0.963 0.967 0.949 0.984
Test
1.00640 - 2-layers GNN - 0.896 0.956 1.038 1.074 1.054 0.912 1.178 0.944
```

## Transferability

The at least 10 movie graph has 1152 nodes, the 50 has 410 and the 150 has 203.

The lab results in the website are not consistent with mine. Their error decreases when the graph increases. The
opposite of mine. Not sure why. Their 2-layers is also better than mine. Maybe this should be revised if I try
regenerate these results with Pytorch Geometric.

Looking at the predictions, it seems like the GNN is most of the time just estimating the average of the target.
Sometimes with a slightly higher value for labels that are large (like 5). Maybe I should check what the solution
provided by the course is estimating.

| Number of layers | Same size (203 nodes) | 2x (410 nodes)   | 5x (1152 nodes) |
|:-----------------|:---------------------:|:----------------:|:---------------:|
| 1                | 0.99791               | 0.99673 (-0.12%) | 0.99824 (0.03%) |
| 2                | 1.01420               | 1.02128 (0.70%)  | 1.02845 (1.41%) |

```txt
$ git checkout lab3-up-to-3.5
$ python lab3_main.py --experiments "GNN" "2-layers GNN" --dataset-glob-pattern "datasets/lab3_similarity_??_150_reviews.pt"
Train
0.99352 - GNN - 0.986 1.001 0.970 0.995 0.983 0.979 1.018 1.016
Test
0.99791 - GNN - 1.066 0.869 1.168 0.961 1.117 1.057 0.833 0.911
0.99673 - 50_reviews - 1.066 0.858 1.170 0.961 1.117 1.057 0.833 0.911
0.99824 - 10_reviews - 1.066 0.869 1.172 0.961 1.117 1.057 0.833 0.911
Train
0.97021 - 2-layers GNN - 0.976 0.945 0.957 0.960 0.953 0.969 0.982 1.020
Test
1.01420 - 2-layers GNN - 1.071 0.879 1.166 0.999 1.082 1.083 0.895 0.937
1.02128 - 50_reviews - 1.080 0.892 1.185 0.984 1.114 1.099 0.888 0.927
1.02845 - 10_reviews - 1.080 0.910 1.185 0.984 1.100 1.108 0.914 0.948
```

## Manual check

I checked the estimations of the reference solution against mine. To check if the model was not just estimating the average. And it seems that both mine and the reference are not very good, they are close to average, the reference does not seems to go down to low-value labels.

For the case below, a estimator that always estimates the average value would produce a MSE of 1.0348, while the GNN has MSE of 1.0323.

```txt
pred         label
[3.90174161, 2.        ],
[4.18058087, 4.        ],
[3.91360401, 4.        ],
[3.67834524, 5.        ],
[3.74828212, 2.        ],
[3.67508965, 4.        ],
[3.82394793, 4.        ],
[3.71462124, 4.        ],
[3.59110503, 3.        ],
[3.7404873 , 3.        ]
```

My GNN is always estimating the same value no matter the input :/. This seems to be highly sensible to hyper
parameters. Changing learning rate, batch size and number of epochs, I could find examples that go closer to the
reference material. And do not estimate always the same output. For the graph of the same size, results could be a bit
better than predicting average for training, but not for test.

Predicting always the average of the target variable yields:

```txt
avg test  - 0.98582 - 1.065 0.823 1.171 0.951 1.093 1.050 0.832 0.901
avg train - 0.99350 - 1.068 0.830 1.171 0.978 1.101 1.059 0.839 0.901
```

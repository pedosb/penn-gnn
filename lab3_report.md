# Comments/results of Lab 2

## 2.2 Linear model vs graph filter

Interesting that the graph filter now is performing better than the linear model, even with the linear model having way more parameters. It seems that the linear model is overfitting the data. Remember that we are training for more epochs now, when compared to lab2.

```txt
$ git checkout lab3-up-to-2.2
$ python lab3_main.py --experiments "Linear" "Filter" --dataset-glob-pattern "datasets/lab3_similarity_??.pt"
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MultiLayerGNN                            --
├─Sequential: 1-1                        --
│    └─GraphFilter: 2-1                  320
│    └─GraphFilter: 2-2                  64
=================================================================
Total params: 384
Trainable params: 384
Non-trainable params: 0
=================================================================
Train
0.04636 - Filter - 0.047 0.048 0.049 0.045 0.042
Test
0.04736 - Filter - 0.043 0.051 0.061 0.029 0.053
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
0.04210 - Linear - 0.042 0.042 0.039 0.043 0.043
Test
0.29309 - Linear - 0.371 0.303 0.308 0.193 0.291
```

# GNN exercises

Labs from [Penn GNN course](https://gnn.seas.upenn.edu/)

This assumes
[`ml-100k.zip`](https://files.grouplens.org/datasets/movielens/ml-100k.zip) is
downloaded under `data`. `curl -o data/ml-100k.zip
https://files.grouplens.org/datasets/movielens/ml-100k.zip`.

## Questions

1. I was expecting to hit performance wise Filter < GNN < multilayer GNN
   (MLGNN). But that did not happen. MLGNN could get better if training for
   longer and using LeakyRELU as activation. It seems that it often gets stuck
   at local minima and does not evolve over training (I have seen sometimes all
   the gradients as 0). However, the solution is comparable to the provided
   code, even thought to match the plots in the website the training instances
   would have to been chosen (it does not seems to represent the expected
   curve).

## Reference results

### Up to Lab 2 - 3.2

> tag: lab2-up-to-3.2

- 5.117169771844705 - Reference filter
- 5.203234351887519 - Reference perceptron
- 5.129227307863615 - My filter
- 5.110793771488497 - My perceptron
- 6.006952027956511 - My MLGNN

## Torch geometric notes

The [GCN from torch
geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv)
applies one multiplication by the GSO and by a coefficient (no nonlinearity).
This could be thought as equivalent to a filter tap, and the self loop takes
care of adding the previous tap.

One observation is that it does not include the first tap, the first layer is
already $S^1h_1x$.

The second layer would be $Sh_2(Sh_1x)$, which I guess could be written as
$h_1h_2S^2x$, if we assume self-loops are added, one of the components would be
previous filter tap, but it would also be scaled by $h_2$ and not just summed
as in the filter.

The closest way to implement I can think of here is to build a custom module,
with multiple GCN layers that are summed together and then the nonlinearity
applied. However that would produce something like $S^1h_1x+S^2h_2h_1x$. I am
not sure how the appearance of $h_1$ on the second tap will influence the
result, I guess that computing gradients will change and probably the
properties of the filter.

It seems that is actually a "feature", each GCN layer is supposed to apply a
single diffusion step and the non-linearity. This is inspired by MLP, where
each layer is processing the feature extraction of the previous layer.

Using SimpleConv...

[GraphCov](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GraphConv.html#torch_geometric.nn.conv.GraphConv)
seems interesting, it would match the filter if $x_i$ was not multiplied by
$W_1$, having it there is training weights on previous filter taps (that were
already weighted by $W_2$).

[SGConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SGConv.html#torch_geometric.nn.conv.SGConv)
is almost what we want, it has a parameter $\mathbf{K}$ that would be the
number of diffusion to apply. However, using it for the filter (as `SGConv(k=1)
+ SGConv(k=2)`) would repeat lots of operations since the computation of tap 2
would redo the computation of tap 1 (that could be reused).

The [SGConv paper](https://arxiv.org/abs/1902.07153) talks about GCN "skipping"
the graph filter step, because it evolved from deep learning, instead of
"naturally" evolving from graph processing, so it skipping the graph filter and
applied the nonlinearity to each tap step, instead of applying it at the end of
the filter (feature extraction).

> TODO: I need to continue reading this paper, from Sec. 3. It could help me
> fix the interpretation of graph filters and GFT.

[SSGConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SSGConv.html#torch_geometric.nn.conv.SSGConv)
goes almost there if $\alpha=0$. But it still applies "filter coefficients"
just at the last tap. In the implementation I cannot see the $\alpha\mathbf(X)$
inside the summation, I was expecting something like [this
comment](https://github.com/allenhaozhu/SSGC/blob/main/ogb-iclr2021/mag/ssgc_embedding.py#L102).
But it seems that this term is used just once. Maybe there is a trick that I am
missing.

Found it :)
[TAGConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TAGConv.html#torch_geometric.nn.conv.TAGConv)

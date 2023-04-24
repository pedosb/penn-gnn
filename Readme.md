# GNN exercises

Labs from [Penn GNN course](https://gnn.seas.upenn.edu/)

## Questions

1. I was expecting to hit performance wise Filter < GNN < multilayer GNN (MLGNN). But that did not happen. MLGNN could
   get better if training for longer and using LeakyRELU as activation. It seems that it often gets stuck at local
   minima and does not evolve over training (I have seen sometimes all the gradients as 0). However, the solution is
   comparable to the provided code, even thought to match the plots in the website the training instances would have to
   been chosen (it does not seems to represent the expected curve).

## Reference results

### Up to Lab 2 - 3.2

> tag: lab2-up-to-3.2

- 5.117169771844705 - Reference filter
- 5.203234351887519 - Reference perceptron
- 5.129227307863615 - My filter
- 5.110793771488497 - My perceptron
- 6.006952027956511 - My MLGNN

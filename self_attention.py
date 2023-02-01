import torch
import torch.nn as nn
from torch.nn import functional as F

# TODO 8 tokens here in a batch and they are currently not talking to each other => couple them(make them communicate)
torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
x.shape

# TODO version1) x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B,T,C)) # TODO initialize to Zero
for b in range(B):
    for t in range(T):
        # TODO easiest way for tokens to communicate is to take an average of all the preceding elements
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev, 0)

# TODO version2) using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))  # TODO tril: get the lower triangular part filled with 1, otherwise 0
wei = wei / wei.sum(1, keepdim=True)    # TODO vector normalize
xbow2 = wei @ x  # TODO matrix multiplication --> (B, T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)
torch.allclose(xbow, xbow2)     # TRUE

# TODO version3) use Softmax
tril = torch.tril(torch.ones(T, T))
# TODO could be think of as affinities between tokens(are not going to be just cst at zero, but data dependent)
wei = torch.zeros((T,T))
# TODO saying that tokens from the past do not communicate with future
wei = wei.masked_fill(tril == 0, float('-inf'))
# TODO Softmax: rescaling by normalization of exponentiated value
wei = F.softmax(wei, dim=-1)
# TODO weighted aggregations of past elements by using matrix multiplication of a lower triangular fashion
xbow3 = wei @ x
torch.allclose(xbow, xbow3)


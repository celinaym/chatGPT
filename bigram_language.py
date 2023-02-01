import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # TODO create Token Embedding Table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)   # TODO matrix, vocab_size x vocab_size

    def forward(self, idx, targets=None):
        # TODO pluck out a row of that embedding table corresponsding to its index
        logits = self.token_embedding_table(idx)    # TODO Pi torch gonna arrange (Batch X Time X Channel(vocab_size))

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape  # TODO documentation accept (BxC), not (BxTxC) => Reshape
            logits = logits.view(B*T, C)    # TODO stretch out 2d arrays to 1d sequence
            targets = targets.view(B*T)     # TODO targets dim currently was BxT(2D) -> B*T(1D)
            loss = F.cross_entropy(logits, targets)     # TODO measures the quality of the logits w.r.t the targets(prediction good?)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # TODO idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # TODO get the predictions
            logits, loss = self(idx)
            # TODO focus only on the last time step
            logits = logits[:, -1, :]   # (BxTxC) becomes (B,C)
            # TODO apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)   # (B,C)
            # TODO sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # single prediction => (B,1)
            # TODO append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
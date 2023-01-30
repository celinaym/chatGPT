import wget
import torch    # TODO we use PyTorch
import numpy
from torch import nn
from torch.nn import functional as F

# TODO 1. download tiny-shakespeare datasets
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
wget.download(url)

# TODO 2. read it in to insepct it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))

# TODO 3. look at the first 1000 characters
print(text[:1000])

# TODO 4. here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# TODO 5. create a mapping from characters to integers(Tokenize)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]     # TODO encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])    # TODO decoder: take a list of integers, output a string
print(encode("hii there"))
print(decode(encode("hii there")))

# TODO 6. encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# TODO 7. split up the data into train and validation sets(first 90% will be train, rest val)
n = int(0.9*len(data))
train_data = data[:n]       # TODO dnt want perfect memorization
val_data = data[n:]

# TODO 8. only work with chunks of the data sets(sample random little chunks out of the td)
block_size = 8      # TODO chunks basically hv max len(block_size)
print(f"chunk is {train_data[:block_size+1]}")
x = train_data[:block_size]
y = train_data[1:block_size+1]  # TODO offset by 1
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

# TODO 9. process multiple processing(chunks) all at the same time
torch.manual_seed(1337)     # TODO generate random number
batch_size = 4      # TODO how many independent sequences will we process in parallel?
block_size = 8      # TODO what is the max context length for predictions?


def get_batch(split):
    # TODO generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    # TODO ix going to be a 4 numbers that are randomly generated between 0 and len(data-block_size)
    ix = torch.randint(len(data) - block_size, size=(batch_size,))
    # TODO first block size characters(stack them up as rows)
    x = torch.stack([data[i:i+block_size] for i in ix])
    # TODO offset by 1
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


xb, yb = get_batch('train')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context} the target: {target}")


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # TODO create Token Embedding Table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)   # TODO matrix, vocab_size x vocab_size

    def forward(self, idx, targets):
        # TODO pluck out a row of that embedding table corresponsding to its index
        logits = self.token_embedding_table(idx)    # TODO Pi torch gonna arrange (Batch X Time X Channel(vocab_size))

        return logits


m = BigramLanguageModel(vocab_size)
out = m(xb, yb)     # TODO call def forward(train_data, val_data)
print(out.shape)










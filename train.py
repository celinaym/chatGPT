import wget
import torch    # TODO we use PyTorch
import numpy
from torch import nn
from torch.nn import functional as F
import bigram_language

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

m = bigram_language.BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)     # TODO call def forward(train_data, val_data)
print(logits.shape)
print(loss)
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))  # garbage value

# TODO 10. create a PyTorch optimizer(lr=learning_rate)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(1000):  # TODO increase number of steps for good results...

    # TODO sample a batch of data
    xb, yb = get_batch('train')

    # TODO evaluate the loss
    logits, loss = m(xb, yb)
    # TODO sets the grad. of all optimized torch.Tensors to zero
    optimizer.zero_grad(set_to_none=True)
    # TODO compute the gradients
    loss.backward()
    # TODO performs a single optimization step(params update). called once the grad. are computed using backward()
    optimizer.step()

print(loss.item())
# TODO max_new_tokens increases -> models more in progress
print(decode(m.generate(idx= torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))









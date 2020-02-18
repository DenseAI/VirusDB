import torch
from reformer_pytorch import ReformerLM

seq_len = 10240

model = ReformerLM(
    num_tokens= 10000,
    dim = 128,
    depth = 2,
    max_seq_len = seq_len,
    heads = 2,
    lsh_dropout = 0.1,
    causal = True,        # auto-regressive or not
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    ff_chunks = 64,      # number of chunks for feedforward layer, make higher if there are memory issues
    weight_tie = False,   # tie parameters of each layer for no memory per additional depth
    attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    use_full_attn = False   # use full self attention, for comparison
).cuda()

print("X: ")
x = torch.randint(0, 10000, (1, seq_len)).long().cuda()

print("Y")
y = model(x)

print("Backward")
y.sum().backward()
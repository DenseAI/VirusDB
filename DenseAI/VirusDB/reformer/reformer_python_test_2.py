import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from reformer_pytorch import Reformer, ReformerLM
from transformers import BertTokenizer, AdamW

import re
import os
from tqdm import tqdm, tqdm_notebook
from glob import glob

import json



tokenizer = BertTokenizer.from_pretrained('/home/huanghaiping/Research/Data/cased_L-12_H-768_A-12/')
tokenizer.max_len = 3072

model = ReformerLM(
    num_tokens=tokenizer.vocab_size,
    dim=100,
    depth=2,
    heads=2,
    max_seq_len=tokenizer.max_len,
    causal=True
)

test = 'Hello, my dog is cute'

tok = tokenizer.encode(test, max_length=tokenizer.max_len, add_special_tokens=True)

print("tok: ", tok)
tokens = []
for ii in range(32):
    tokens.append(tok)

# tok = np.array(tokens)
tok = torch.tensor(tok, dtype=torch.long)




print(tokens)

# tokenizer.decode(tok)


def mask_tokens(inputs: torch.Tensor, tokenizer, mlm_probability=0.15, pad=True):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # mlm_probability defaults to 0.15 in Bert
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]

    # print("special_tokens_mask: ", special_tokens_mask)
    # print(torch.tensor(special_tokens_mask).byte())
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask).byte(), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).byte()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).byte() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).byte() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    inputs_list = []
    labels_list = []

    if pad:
        input_pads = tokenizer.max_len - inputs.shape[-1]
        label_pads = tokenizer.max_len - labels.shape[-1]


        inputs = F.pad(inputs, pad=(0, input_pads), value=tokenizer.pad_token_id)
        labels = F.pad(labels, pad=(0, label_pads), value=tokenizer.pad_token_id)

        # for ii in range(32):
        #     inputs_list.append(inputs)
        #     labels_list.append(labels)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    # print("inputs: ", inputs)
    # torch.tensor()
    return inputs, labels


inputs, labels = mask_tokens(tok.unsqueeze(0), tokenizer, pad=True)

# tokenizer.decode(inputs.squeeze(0))
#
# tokenizer.decode(labels.squeeze(0))

pred = model(inputs)
pred.shape

tokenizer.decode(torch.argmax(pred, dim=-1).squeeze(0))

loss_fn = nn.CrossEntropyLoss()  #

masked_lm_loss = loss_fn(pred.view(-1, tokenizer.vocab_size), labels.view(-1))
masked_lm_loss

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
total_loss = 0.0
model.train()

model.to(device)
inputs = inputs.to(device)
labels = labels.to(device)

loss = []
optimizer = AdamW(params=model.parameters())

for _ in tqdm(range(100000)):
    pred = model(inputs)
    mlm_loss = loss_fn(pred.view(-1, tokenizer.vocab_size), labels.view(-1))

    total_loss += mlm_loss.item()
    loss.append(mlm_loss.item())

    mlm_loss.backward()
    optimizer.step()
    model.zero_grad()

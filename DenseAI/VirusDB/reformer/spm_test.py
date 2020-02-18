
from tensorflow.compat.v1.io.gfile import GFile
import gin
import os
import jax
import trax
from trax.supervised import inputs

import numpy as onp
import jax.numpy as np

from scipy.special import softmax

from sentencepiece import SentencePieceProcessor


import sentencepiece as spm


# Configure hyperparameters.
gin.parse_config("""
import trax.layers
import trax.models
import trax.optimizers
import trax.supervised.inputs
import trax.supervised.trainer_lib

# Parameters that will vary between experiments:
# ==============================================================================
train.model = @trax.models.ReformerLM
attn_type = @TimeBinCausalAttention
share_qk = False  # LSHCausalAttention ignores this flag and always shares q & k
attn_kv = 64
n_layers = 6

share_qk = False  # LSHCausalAttention ignores this flag and always shares q & k
n_heads = 2
attn_kv = 64
dropout = 0.05

n_tokens = 524288

# MemoryEfficientCausalAttention: full attention
# (no hparams to vary between experiments)

# TimeBinCausalAttention: attend to nearby items
TimeBinCausalAttention.bin_length = 512



# Parameters for MultifactorSchedule:
# ==============================================================================
# 0.03125 ~= 1024^-0.5 = d_model^-0.5
MultifactorSchedule.constant = 0.03125
MultifactorSchedule.factors = 'constant * linear_warmup * rsqrt_decay'
MultifactorSchedule.warmup_steps = 2000

# Parameters for Adam:
# ==============================================================================
Adam.weight_decay_rate=0.0
Adam.b1 = 0.9
Adam.b2 = 0.98
Adam.eps = 1e-9

# Parameters for train:
# ==============================================================================
train.eval_frequency = 20
train.eval_steps = 1
train.inputs = @trax.supervised.inputs.inputs
# train.model: see top
train.optimizer = @trax.optimizers.Adam
train.steps = 100000
train.save_graphs = False
train.has_weights = True

# Parameters for TimeBinCausalAttention:
# ==============================================================================
TimeBinCausalAttention.dropout = 0.1
# TimeBinCausalAttention.n_bins: see top
TimeBinCausalAttention.share_qk = %share_qk

# Parameters for ReformerLM:
# ==============================================================================
ReformerLM.attention_type = %attn_type
ReformerLM.d_attention_key = %attn_kv
ReformerLM.d_attention_value = %attn_kv
ReformerLM.d_model = 256
ReformerLM.d_ff = 512
ReformerLM.dropout = %dropout
ReformerLM.ff_activation = @trax.layers.Relu
ReformerLM.max_len = %n_tokens
ReformerLM.mode = 'train'
ReformerLM.n_heads = %n_heads
ReformerLM.n_layers = %n_layers
ReformerLM.vocab_size = 320
ReformerLM.share_qk = %share_qk
ReformerLM.axial_pos_shape = (512, 1024)
ReformerLM.d_axial_pos_embs= (64, 192)

""")



with open('2554.txt') as f:
  text = f.read()


# The file read above includes metadata and licensing information.
# For training our language model, we will only use the actual novel text.
start = text.find('CRIME AND PUNISHMENT')  # skip header
start = text.find('CRIME AND PUNISHMENT', start + 1)  # skip header
start = text.find('CRIME AND PUNISHMENT', start + 1)  # skip translator preface
end = text.rfind('End of Project')  # skip extra text at the end
text = text[start:end].strip()


TOKENIZER = SentencePieceProcessor()
TOKENIZER.load('test_model.model')

# print("Text: ", text)

# Tokenize
IDS = TOKENIZER.EncodeAsIds(text)
IDS = onp.asarray(IDS, dtype=onp.int32)
PAD_AMOUNT = 512 * 1024 - len(IDS)
print("Number of tokens:", IDS.shape[0])




# Set up the data pipeline.
# Set up the data pipeline.
def my_inputs(n_devices):
  while True:
    inputs = []
    mask = []
    pad_amounts = onp.random.choice(PAD_AMOUNT, n_devices)
    for i in range(n_devices):
      inputs.append(onp.pad(IDS, (pad_amounts[i], PAD_AMOUNT - pad_amounts[i]),
                            mode='constant'))
      mask.append(onp.pad(onp.ones_like(IDS, dtype=onp.float32),
                          (pad_amounts[i], PAD_AMOUNT - pad_amounts[i]),
                          mode='constant'))
    inputs = onp.stack(inputs)
    mask = onp.stack(mask)
    yield (inputs, inputs, mask)

print("(device count, tokens per device) = ",
      next(my_inputs(trax.math.device_count()))[0].shape)


# PAD_AMOUNT = 512 * 1024 - len(IDS)
output_dir = os.path.expanduser('~/train_dir/')
# !rm -f ~/train_dir/model.pkl  # Remove old model
trainer = trax.supervised.Trainer(
    model=trax.models.ReformerLM,
    loss_fn=trax.layers.CrossEntropyLoss,
    optimizer=trax.optimizers.Adam,
    lr_schedule=trax.lr.MultifactorSchedule,
    inputs=trax.supervised.inputs.Inputs(my_inputs),
    output_dir=output_dir,
    has_weights=True)


print(trainer)

# Run one training step, to make sure the model fits in memory.
# The first time trainer.train_epoch is called, it will JIT the entire network
# architecture, which takes around 2 minutes. The JIT-compiled model is saved
# so subsequent runs will be much faster than the first.
trainer.train_epoch(n_steps=1, n_eval_steps=1)
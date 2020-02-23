import os
import numpy as np

import trax
import jax
import gin
from trax import layers as tl

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["JAX_PLATFORM_NAME"] = 'gpu'

from jax.config import config
# config.FLAGS.jax_xla_backend = "tpu_driver"

#jax_platform_name
# jax.config.update('jax_platform_name', 'CUDA')

max_sel_len = 3072

length = max_sel_len
vocab_size = 32
batch_size = 32

w_length = (length // 2) - 1
w = np.random.randint(low=1, high=vocab_size - 1,
                      size=(batch_size, w_length))
zero = np.zeros([batch_size, 1], np.int32)
loss_weights = np.concatenate([np.zeros((batch_size, w_length)),
                               np.ones((batch_size, w_length + 2))], axis=1)


def copy_task(batch_size, vocab_size, length):
    """This task is to copy a random string w, so the input is 0w0w."""
    while True:
        assert length % 2 == 0
        # w_length = (length // 2) - 1
        # w = np.random.randint(low=1, high=vocab_size - 1,
        #                       size=(batch_size, w_length))
        # zero = np.zeros([batch_size, 1], np.int32)
        # loss_weights = np.concatenate([np.zeros((batch_size, w_length)),
        #                                np.ones((batch_size, w_length + 2))], axis=1)
        x = np.concatenate([zero, w, zero, w], axis=1)
        yield (x, x, loss_weights)  # Here inputs and targets are the same.


copy_inputs = trax.supervised.Inputs(lambda _: copy_task(32, 32, max_sel_len))

# Peek into the inputs.
data_stream = copy_inputs.train_stream(1)
inputs, targets, mask = next(data_stream)
print("Inputs[0]:  %s" % str(inputs[0]))
print("Targets[0]: %s" % str(targets[0]))
print("Mask[0]:    %s" % str(mask[0]))


gin.parse_config("""
import trax.layers
import trax.models
import trax.optimizers
import trax.supervised.inputs
import trax.supervised.trainer_lib

# Parameters for LSHCausalAttention:
# ==============================================================================
# LSHCausalAttention.allow_duplicate_attention = False
# LSHCausalAttention.attend_across_buckets = True
# LSHCausalAttention.rehash_each_round = True
# LSHCausalAttention.data_rotation = False
# # LSHCausalAttention.n_bins = 256
# LSHCausalAttention.n_buckets = 256
# # LSHCausalAttention.factorize_hash = [64, 128]
# LSHCausalAttention.n_hashes = 1
# LSHCausalAttention.one_rng = False
# LSHCausalAttention.hard_k = 0
LSHCausalAttention.dropout = 0.0
# LSHCausalAttention.drop_for_hash_rate = 0.0
LSHCausalAttention.max_len_for_inference = 3072
# LSHCausalAttention.bucket_capacity_for_inference = 64

""")

# Transformer LM
def tiny_transformer_lm(mode):
    return trax.models.ReformerLM(  # You can try trax_models.ReformerLM too.  TransformerLM
        d_model=32, d_ff=64, n_layers=2, vocab_size=32, share_qk=True, attention_type= tl.LSHCausalAttention, mode=mode, max_len=max_sel_len)


# Train tiny model with Trainer.
output_dir = os.path.expanduser('~/train_dir/')
# !rm -f ~/train_dir/model.pkl  # Remove old model.
trainer = trax.supervised.Trainer(
    model=tiny_transformer_lm,
    loss_fn=trax.layers.CrossEntropyLoss,
    optimizer=trax.optimizers.Adafactor,  # Change optimizer params here.
    lr_schedule=trax.lr.MultifactorSchedule,  # Change lr schedule here.
    inputs=copy_inputs,
    output_dir=output_dir,
    has_weights=True)  # Because we have loss mask, this API may change.

# Train for 3 epochs each consisting of 500 train batches, eval on 2 batches.
n_epochs = 10
train_steps = 312   #5
eval_steps = 2

print("Training: ")
for _ in range(n_epochs):
    trainer.train_epoch(train_steps, eval_steps)

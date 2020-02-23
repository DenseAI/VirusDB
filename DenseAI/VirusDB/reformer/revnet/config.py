# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reversible residual network compatible with eager execution.

Configuration in format of tf.contrib.training.HParams.
Supports Transformer, Reformer.

Reference [The Reversible Residual Network: Backpropagation
Without Storing Activations](https://arxiv.org/pdf/1707.04585.pdf)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_hparams_reformer():
  """
  Reformer configurations.


  Args:
    vocab_size: int: vocab size
    d_model: int:  depth of *each half* of the two-part features
    d_ff: int: depth of feed-forward layer
    d_attention_key: int: depth of key vector for each attention head
    d_attention_value: int: depth of value vector for each attention head
    n_layers: int: number of decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    n_chunks: int: number of chunks (must match input pipeline)
    n_attention_chunks: int: number of chunks for attention
    attention_type: class: attention class to use, such as DotProductAttention.
    share_qk: bool, whether to share queries and keys.
    axial_pos_shape: tuple of ints: input shape to use for the axial position
      encoding. If unset, axial position encoding is disabled.
    d_axial_pos_embs: tuple of ints: depth of position embedding for each axis.
      Tuple length must match axial_pos_shape, and values must sum to d_model.
    ff_activation: the non-linearity in feed-forward layer
    ff_use_sru: int; if > 0, we use this many SRU layers instead of feed-forward
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    mode: str: 'train', 'eval', or 'predict'
  """

  config = tf.contrib.training.HParams()
  config.add_hparam("vocab_size", 10000)
  config.add_hparam("d_model", 512)
  config.add_hparam("d_ff", 2048)
  config.add_hparam("d_attention_key", 64)
  config.add_hparam("d_attention_value", 64)
  config.add_hparam("n_layers", 6)
  config.add_hparam("n_heads", 8)
  config.add_hparam("dropout", 0.1)
  config.add_hparam("max_len", 2048)
  config.add_hparam("n_chunks", 0)
  config.add_hparam("n_attention_chunks", 1)
  config.add_hparam("attention_type", None)
  config.add_hparam("share_qk", False)
  config.add_hparam("axial_pos_shape", None)
  config.add_hparam("d_axial_pos_embs", None)
  config.add_hparam("ff_activation", None)
  config.add_hparam("ff_use_sru", 0)
  config.add_hparam("ff_chunk_size", 0)
  config.add_hparam("mode", 'train')

  # Training details
  config.add_hparam("weight_decay", 1e-4)
  config.add_hparam("momentum", .9)
  config.add_hparam("lr_decay_steps", [160000, 320000, 480000])
  config.add_hparam("lr_list", [1e-1, 1e-2, 1e-3, 1e-4])

  config.add_hparam("seed", 1234)
  config.add_hparam("shuffle", True)
  config.add_hparam("log_every", 500)
  config.add_hparam("save_every", 500)
  config.add_hparam("dtype", tf.float32)
  config.add_hparam("eval_batch_size", 256)



  return config



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

Code for main model.

Reference [The Reversible Residual Network: Backpropagation
Without Storing Activations](https://arxiv.org/pdf/1707.04585.pdf)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from tensorflow.keras.layers import
from DenseAI.VirusDB.reformer.revnet import blocks
from DenseAI.VirusDB.reformer.revnet import config
from DenseAI.VirusDB.reformer.attention import *



class InitBlock(tf.keras.Model):
    """Initial block of RevNet."""

    def __init__(self, config: tf.contrib.training.HParams):
        """Initialization.

        Args:
          config: tf.contrib.training.HParams object; specifies hyperparameters
        """
        super(InitBlock, self).__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.d_model, name='embeddings')

    def call(self, x, training=True):
        x = self.embedding_layer(x)
        return (x, x)  # tf.split(net, num_or_size_splits=2, axis=self.axis)


class FinalBlock(tf.keras.Model):
    """Final block of Reformer."""

    def __init__(self, config):
        """Initialization.

        Args:
          config: tf.contrib.training.HParams object; specifies hyperparameters

        Raises:
          ValueError: Unsupported data format
        """
        super(FinalBlock, self).__init__()
        self.config = config
        self.axis = 1
        self.flatten = tf.keras.layers.Flatten()
        self.activation = tf.keras.layers.Activation("relu")
        self.dense = tf.keras.layers.Dense(self.config.n_classes)

    def call(self, x, training=True):

        print("x: ", x)

        net = tf.concat(x, axis=self.axis)
        net = self.flatten(net)
        net = self.activation(net)
        net = self.dense(net)
        return net




class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_head_count:int = 2, d_model:int = 2, d_point_wise_ff: int = 1024, dropout_prob:float=0.0):
        super(EncoderLayer, self).__init__()

        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.multi_head_attention = MultiHeadAttention(attention_head_count, d_model)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
            d_point_wise_ff,
            d_model
        )
        self.dropout_2 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask, training):
        output, attention = self.multi_head_attention(inputs, inputs, inputs, mask)
        # output = self.dropout_1(output, training=training)
        # output = self.layer_norm_1(tf.add(inputs, output))  # residual network
        #
        # output = self.position_wise_feed_forward_layer(output)
        # output = self.dropout_2(output, training=training)
        # output = self.layer_norm_2(tf.add(inputs, output))  # residual network

        print("output： ", output)
        print("attention： ", attention)
        return output, attention


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob):
        super(DecoderLayer, self).__init__()

        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.masked_multi_head_attention = MultiHeadAttention(attention_head_count, d_model)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.encoder_decoder_attention = MultiHeadAttention(attention_head_count, d_model)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
            d_point_wise_ff,
            d_model
        )
        self.dropout_3 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, decoder_inputs, encoder_output, look_ahead_mask, padding_mask, training):


        output, attention_1 = self.masked_multi_head_attention(
            decoder_inputs,
            decoder_inputs,
            decoder_inputs,
            look_ahead_mask
        )
        # output = self.dropout_1(output, training=training)
        # query = self.layer_norm_1(tf.add(decoder_inputs, output))  # residual network
        # output, attention_2 = self.encoder_decoder_attention(
        #     query,
        #     encoder_output,
        #     encoder_output,
        #     padding_mask
        # )
        # output = self.dropout_2(output, training=training)
        # encoder_decoder_attention_output = self.layer_norm_2(tf.add(output, query))
        #
        # output = self.position_wise_feed_forward_layer(encoder_decoder_attention_output)
        # output = self.dropout_3(output, training=training)
        # output = self.layer_norm_3(tf.add(encoder_decoder_attention_output, output))  # residual network
        print("attention_1: ", attention_1)
        return attention_1 #, attention_1, attention_2


class PositionWiseFeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_point_wise_ff, d_model):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.w_1 = tf.keras.layers.Dense(d_point_wise_ff)
        self.w_2 = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        inputs = self.w_1(inputs)
        inputs = tf.nn.relu(inputs)
        return self.w_2(inputs)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model):
        super(MultiHeadAttention, self).__init__()

        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model

        if d_model % attention_head_count != 0:
            raise ValueError(
                "d_model({}) % attention_head_count({}) is not zero.d_model must be multiple of attention_head_count.".format(
                    d_model, attention_head_count
                )
            )

        self.d_h = d_model // attention_head_count

        self.w_query = tf.keras.layers.Dense(d_model)
        self.w_key = tf.keras.layers.Dense(d_model)
        self.w_value = tf.keras.layers.Dense(d_model)

        self.scaled_dot_product = ScaledDotProductAttention(self.d_h)

        self.ff = tf.keras.layers.Dense(d_model)

    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]

        query = self.w_query(query)
        key = self.w_key(key)
        value = self.w_value(value)

        query = self.split_head(query, batch_size)
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)

        output, attention = self.scaled_dot_product(query, key, value, mask)
        output = self.concat_head(output, batch_size)

        return self.ff(output), attention

    def split_head(self, tensor, batch_size):
        # inputs tensor: (batch_size, seq_len, d_model)
        return tf.transpose(
            tf.reshape(
                tensor,
                (batch_size, -1, self.attention_head_count, self.d_h)
                # tensor: (batch_size, seq_len_splited, attention_head_count, d_h)
            ),
            [0, 2, 1, 3]
            # tensor: (batch_size, attention_head_count, seq_len_splited, d_h)
        )

    def concat_head(self, tensor, batch_size):
        return tf.reshape(
            tf.transpose(tensor, [0, 2, 1, 3]),
            (batch_size, -1, self.attention_head_count * self.d_h)
        )


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_h):
        super(ScaledDotProductAttention, self).__init__()
        self.d_h = d_h

    def call(self, query, key, value, mask=None):
        matmul_q_and_transposed_k = tf.matmul(query, key, transpose_b=True)
        scale = tf.sqrt(tf.cast(self.d_h, dtype=tf.float32))
        scaled_attention_score = matmul_q_and_transposed_k / scale
        if mask is not None:
            scaled_attention_score += (mask * -1e9)

        attention_weight = tf.nn.softmax(scaled_attention_score, axis=-1)

        return tf.matmul(attention_weight, value), attention_weight


class Embeddinglayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        # model hyper parameter variables
        super(Embeddinglayer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

    def call(self, sequences):
        max_sequence_len = sequences.shape[1]
        output = self.embedding(sequences) * tf.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        output += self.positional_encoding(max_sequence_len)

        return output

    def positional_encoding(self, max_len):
        pos = np.expand_dims(np.arange(0, max_len), axis=1)
        index = np.expand_dims(np.arange(0, self.d_model), axis=0)

        pe = self.angle(pos, index)

        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        pe = np.expand_dims(pe, axis=0)
        return tf.cast(pe, dtype=tf.float32)

    def angle(self, pos, index):
        return pos / np.power(10000, (index - index % 2) / np.float32(self.d_model))

# Ideally, the following should be wrapped in `tf.keras.Sequential`, however
# there are subtle issues with its placeholder insertion policy and batch norm
class F_Block(tf.keras.Model):
    """Single bottleneck residual inner function contained in _Resdual.

    Corresponds to the `F`/`G` functions in the paper.
    Suitable for training on ImageNet dataset.
    """

    def __init__(self,
                 num_heads: int = 2,
                 fused=True,
                 dtype=tf.float32):
        """
        Initialization.

        Args:
          fused: use fused batch normalization if True
          dtype: float16, float32, or float64
        """
        super(F_Block, self).__init__()
        self.axis = 1
        self.num_heads = num_heads
        self.residual_dropout = 0.0
        self.attention_dropout = 0.0
        self.use_universal_transformer = False
        # self.act_layer = TransformerACT(name='adaptive_computation_time')
        # self.transformer_block = TransformerBlock(
        #     name='transformer',
        #     num_heads=self.num_heads,
        #     residual_dropout=self.residual_dropout,
        #     attention_dropout=self.attention_dropout,
        #     # Allow bi-directional attention
        #     use_masking=False)
        # self.coordinate_embedding_layer = TransformerCoordinateEmbedding(
        #     self.max_depth if self.use_universal_transformer else 1,
        #     name='coordinate_embedding')

        self.batch_norm_1 = tf.keras.layers.BatchNormalization(
            axis=self.axis, fused=fused, dtype=dtype)

        # self.ff = PositionWiseFeedForwardLayer()
        self.encoder = EncoderLayer()



    def call(self, x, training=True):
        net = x
        # next_step_input = self.coordinate_embedding_layer(next_step_input)
        #
        # print("next_step_input: ", next_step_input)
        #
        # next_step_input = self.transformer_block(next_step_input)
        # next_step_input, act_output = self.act_layer(next_step_input)
        #
        # self.act_layer.finalize()

        # net = self.batch_norm_1(net, training=training)
        # net = self.ff(net)
        self.encoder(net, None, training=training)

        return net


class G_Block(tf.keras.Model):
    """Single residual inner function contained in _ResdualBlock.

    Corresponds to the `F`/`G` functions in the paper.
    """

    def __init__(self,
                 # filters,
                 # strides,
                 input_shape=(10,),
                 batch_norm_first=True,
                 data_format="channels_first",
                 fused=True,
                 dtype=tf.float32):
        """Initialization.

        Args:
          filters: output filter size
          strides: length 2 list/tuple of integers for height and width strides
          input_shape: length 3 list/tuple of integers
          batch_norm_first: whether to apply activation and batch norm before conv
          data_format: tensor data format, "NCHW"/"NHWC"
          fused: use fused batch normalization if True
          dtype: float16, float32, or float64
        """
        super(G_Block, self).__init__()
        axis = 1 if data_format == "channels_first" else 3
        # if batch_norm_first:
        #     self.batch_norm_0 = tf.keras.layers.BatchNormalization(
        #         axis=axis, input_shape=input_shape, fused=fused, dtype=dtype)
        #

        self.batch_norm_1 = tf.keras.layers.BatchNormalization(
            axis=axis, fused=fused, dtype=dtype)
        # self.conv2d_2 = tf.keras.layers.Conv2D(
        #     filters=filters,
        #     kernel_size=3,
        #     strides=(1, 1),
        #     data_format=data_format,
        #     use_bias=False,
        #     padding="SAME",
        #     dtype=dtype)

        # self.batch_norm_first = batch_norm_first

    def call(self, x, training=True):
        net = x
        # if self.batch_norm_first:
        #     net = self.batch_norm_0(net, training=training)
        #     net = tf.nn.relu(net)
        # net = self.conv2d_1(net)

        net = self.batch_norm_1(net, training=training)
        # net = tf.nn.relu(net)
        # net = self.conv2d_2(net)

        return net



class Reformer(tf.keras.Model):
    """Reformer that depends on all the blocks."""

    def __init__(self, config):
        """
        Initialize Reformer with building blocks.
        Args:
          config: tf.contrib.training.HParams object; specifies hyperparameters
        """
        super(Reformer, self).__init__()
        self.config = config

        self._init_block = InitBlock(config=self.config)
        self._final_block = FinalBlock(config=self.config)

        self._block_list = self._construct_intermediate_blocks()
        self._moving_average_variables = []

    def _construct_intermediate_blocks(self):
        # Precompute input shape after initial block
        # stride = self.config.init_stride

        # Aggregate intermediate blocks
        block_list = tf.contrib.checkpoint.List()
        for i in range(self.config.n_layers):
            # RevBlock configurations
            # n_res = self.config.n_res[i]
            f_block = F_Block()
            g_block = G_Block()

            # Add block
            rev_block = blocks.ReversibleSequence(
                f_block,
                g_block,
                # batch_norm_first=(i != 0),  # Only skip on first block
                # data_format=self.config.data_format,
                # bottleneck=self.config.bottleneck,
                # fused=self.config.fused,
                dtype=self.config.dtype)
            block_list.append(rev_block)

        return block_list

    def call(self, inputs, training=True):
        """Forward pass."""

        if training:
            saved_hidden = [inputs]

        h = self._init_block(inputs, training=training)
        if training:
            saved_hidden.append(h)

        for block in self._block_list:
            h = block(h, training=training)
            if training:
                saved_hidden.append(h)

        logits = self._final_block(h, training=training)

        return (logits, saved_hidden) if training else (logits, None)

    def compute_loss(self, logits, labels):
        """Compute cross entropy loss."""
        if self.config.dtype == tf.float32 or self.config.dtype == tf.float16:
            cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
        else:
            # `sparse_softmax_cross_entropy_with_logits` does not have a GPU kernel
            # for float64, int32 pairs
            labels = tf.one_hot(
                labels, depth=self.config.n_classes, axis=1, dtype=self.config.dtype)
            cross_ent = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)

        return tf.reduce_mean(cross_ent)

    def compute_gradients(self, saved_hidden, labels, training=True, l2_reg=True):
        """
        Manually computes gradients.

        This method silently updates the running averages of batch normalization.

        Args:
          saved_hidden: List of hidden states Tensors
          labels: One-hot labels for classification
          training: Use the mini-batch stats in batch norm if set to True
          l2_reg: Apply l2 regularization

        Returns:
          A tuple with the first entry being a list of all gradients and the second
          being the loss
        """

        def _defunable_pop(l):
            """Functional style list pop that works with `tfe.defun`."""
            t, l = l[-1], l[:-1]
            return t, l

        # Backprop through last block
        x = saved_hidden[-1]
        with tf.GradientTape() as tape:
            tape.watch(x)

            print("x: ", x)
            logits = self._final_block(x, training=training)
            loss = self.compute_loss(logits, labels)
        grads_combined = tape.gradient(loss,
                                       [x] + self._final_block.trainable_variables)
        dy, final_grads = grads_combined[0], grads_combined[1:]

        # Backprop through intermediate blocks
        intermediate_grads = []
        for block in reversed(self._block_list):
            y, saved_hidden = _defunable_pop(saved_hidden)
            x = saved_hidden[-1]
            dy, grads = block.backward_grads(x, y, dy, training=training)
            intermediate_grads = grads + intermediate_grads

        # Backprop through first block
        _, saved_hidden = _defunable_pop(saved_hidden)
        x, saved_hidden = _defunable_pop(saved_hidden)
        assert not saved_hidden
        with tf.GradientTape() as tape:
            y = self._init_block(x, training=training)
        init_grads = tape.gradient(
            y, self._init_block.trainable_variables, output_gradients=dy)

        # Ordering match up with `model.trainable_variables`
        grads_all = init_grads + final_grads + intermediate_grads
        if l2_reg:
            grads_all = self._apply_weight_decay(grads_all)

        return grads_all, loss

    def _apply_weight_decay(self, grads):
        """Update gradients to reflect weight decay."""
        return [
            g + self.config.weight_decay * v if v.name.endswith("kernel:0") else g
            for g, v in zip(grads, self.trainable_variables)
        ]

    def get_moving_stats(self):
        """Get moving averages of batch normalization."""
        device = "/gpu:0" if tf.test.is_gpu_available() else "/cpu:0"
        with tf.device(device):
            return [v.read_value() for v in self.moving_average_variables]

    def restore_moving_stats(self, values):
        """Restore moving averages of batch normalization."""
        device = "/gpu:0" if tf.test.is_gpu_available() else "/cpu:0"
        with tf.device(device):
            for var_, val in zip(self.moving_average_variables, values):
                var_.assign(val)

    @property
    def moving_average_variables(self):
        """Get all variables that are batch norm moving averages."""
        def _is_moving_avg(v):
            n = v.name
            return n.endswith("moving_mean:0") or n.endswith("moving_variance:0")

        if not self._moving_average_variables:
            self._moving_average_variables = filter(_is_moving_avg, self.variables)

        return self._moving_average_variables


if __name__ == '__main__':

    # from tensorflow.contrib.eager.python.examples.revnet import config as config_
    config_ = config

    """Test model training in graph mode."""
    with tf.Graph().as_default():
        configure = config_.get_hparams_reformer()
        configure.add_hparam("n_classes", 10)
        configure.add_hparam("batch_size", 32)

        # configure.add_hparam("batch_size", 32)
        configure.add_hparam("num_train_size", 100000)
        configure.add_hparam("max_train_iter", 100000)

        # configure.add_hparam("iters_per_epoch", config.num_train_size // config.batch_size)
        # configure.add_hparam("epochs", config.max_train_iter // config.iters_per_epoch)


        x = tf.random_uniform(
            shape=(configure.batch_size,) + (10,), minval=1, maxval=100, dtype="int32")  # configure.input_shape)
        # shape = (configure.batch_size,) + configure.input_shape)
        t = tf.random_uniform(
            shape=(configure.batch_size,),
            minval=0,
            maxval=configure.n_classes,
            dtype=tf.int32)

        print("x: ", x, str(x))
        print("t: ", t)
        global_step = tf.Variable(0., trainable=False)
        model = Reformer(config=configure)

        _, saved_hidden = model(x)
        grads, _ = model.compute_gradients(saved_hidden=saved_hidden, labels=t)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.apply_gradients(
            zip(grads, model.trainable_variables), global_step=global_step)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(1):
                sess.run(train_op)

        model.summary()

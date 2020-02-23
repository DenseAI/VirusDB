

import math
import tensorflow as tf
from keras.layers import Dense
import numpy as np
from keras import backend as K

class FeedForward(tf.keras.Model):
    def __init__(self, emb, mult=4):
        super().__init__()
        self.emb = emb
        self.proj_in = Dense(emb * mult)
        self.proj_out = Dense(emb)

    def call(self, inputs):
        inputs = self.proj_in(inputs)
        inputs = tf.keras.activations.relu(inputs)
        inputs = self.proj_out(inputs)
        return inputs


class _BaseMultiHeadAttention(tf.keras.layers.Layer):
    """
    Base class for two types of Multi-head attention layers:
    Self-attention and its more general form used in decoders (the one which
    takes values and keys from the encoder).
    """
    def __init__(self, num_heads: int, use_masking: bool,
                 dropout: float = 0.0,
                 compression_window_size: int = None,
                 **kwargs):
        """
        :param num_heads: number of attention heads
        :param use_masking: when True, forbids the attention to see the further
          elements in the sequence (particularly important in language
          modelling).
        :param dropout: dropout that should be applied to the attention
          (after the softmax).
        :param compression_window_size: an integer value >= 1 controlling
          how much we should compress the attention. For more details,
          read about memory-compressed self-attention in
          "Generating Wikipedia by summarizing long sequences"
          (https://arxiv.org/pdf/1801.10198.pdf).
        :param kwargs: any extra arguments typical for a Keras layer,
          such as name, etc.
        """
        self.num_heads = num_heads
        self.use_masking = use_masking
        self.dropout = dropout
        if (compression_window_size is not None
                and compression_window_size <= 0):
            assert ValueError(
                f"Too small compression window ({compression_window_size})")
        self.compression_window_size = compression_window_size
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['num_heads'] = self.num_heads
        config['use_masking'] = self.use_masking
        config['dropout'] = self.dropout
        config['compression_window_size'] = self.compression_window_size
        return config

    # noinspection PyAttributeOutsideInit
    def build_output_params(self, d_model):
        self.output_weights = self.add_weight(
            name='output_weights',
            shape=(d_model, d_model),
            initializer='glorot_uniform',
            trainable=True)
        if self.compression_window_size is not None:
            self.k_conv_kernel = self.add_weight(
                name='k_conv_kernel',
                shape=(self.compression_window_size,
                       d_model // self.num_heads,
                       d_model // self.num_heads),
                initializer='glorot_uniform',
                trainable=True)
            self.k_conv_bias = self.add_weight(
                name='k_conv_bias',
                shape=(d_model // self.num_heads,),
                initializer='zeros',
                trainable=True)
            self.v_conv_kernel = self.add_weight(
                name='v_conv_kernel',
                shape=(self.compression_window_size,
                       d_model // self.num_heads,
                       d_model // self.num_heads),
                initializer='glorot_uniform',
                trainable=True)
            self.v_conv_bias = self.add_weight(
                name='v_conv_bias',
                shape=(d_model // self.num_heads,),
                initializer='zeros',
                trainable=True)

    def validate_model_dimensionality(self, d_model: int):
        if d_model % self.num_heads != 0:
            raise ValueError(
                f'The size of the last dimension of the input '
                f'({d_model}) must be evenly divisible by the number'
                f'of the attention heads {self.num_heads}')

    def attention(self, pre_q, pre_v, pre_k, out_seq_len: int, d_model: int,
                  training=None):
        """
        Calculates the output of the attention once the affine transformations
        of the inputs are done. Here's the shapes of the arguments:
        :param pre_q: (batch_size, q_seq_len, num_heads, d_model // num_heads)
        :param pre_v: (batch_size, v_seq_len, num_heads, d_model // num_heads)
        :param pre_k: (batch_size, k_seq_len, num_heads, d_model // num_heads)
        :param out_seq_len: the length of the output sequence
        :param d_model: dimensionality of the model (by the paper)
        :param training: Passed by Keras. Should not be defined manually.
          Optional scalar tensor indicating if we're in training
          or inference phase.
        """
        # shaping Q and V into (batch_size, num_heads, seq_len, d_model//heads)
        q = K.permute_dimensions(pre_q, [0, 2, 1, 3])
        v = K.permute_dimensions(pre_v, [0, 2, 1, 3])

        if self.compression_window_size is None:
            k_transposed = K.permute_dimensions(pre_k, [0, 2, 3, 1])
        else:
            # Memory-compressed attention described in paper
            # "Generating Wikipedia by Summarizing Long Sequences"
            # (https://arxiv.org/pdf/1801.10198.pdf)
            # It compresses keys and values using 1D-convolution which reduces
            # the size of Q * K_transposed from roughly seq_len^2
            # to convoluted_seq_len^2. If we use strided convolution with
            # window size = 3 and stride = 3, memory requirements of such
            # memory-compressed attention will be 9 times smaller than
            # that of the original version.
            if self.use_masking:
                raise NotImplementedError(
                    "Masked memory-compressed attention has not "
                    "been implemented yet")
            k = K.permute_dimensions(pre_k, [0, 2, 1, 3])
            k, v = [
                K.reshape(
                    # Step 3: Return the result to its original dimensions
                    # (batch_size, num_heads, seq_len, d_model//heads)
                    K.bias_add(
                        # Step 3: ... and add bias
                        K.conv1d(
                            # Step 2: we "compress" K and V using strided conv
                            K.reshape(
                                # Step 1: we reshape K and V to
                                # (batch + num_heads,  seq_len, d_model//heads)
                                item,
                                (-1,
                                 K.int_shape(item)[-2],
                                 d_model // self.num_heads)),
                            kernel,
                            strides=self.compression_window_size,
                            padding='valid', data_format='channels_last'),
                        bias,
                        data_format='channels_last'),
                    # new shape
                    K.concatenate([
                        K.shape(item)[:2],
                        [-1, d_model // self.num_heads]]))
                for item, kernel, bias in (
                    (k, self.k_conv_kernel, self.k_conv_bias),
                    (v, self.v_conv_kernel, self.v_conv_bias))]
            k_transposed = K.permute_dimensions(k, [0, 1, 3, 2])
        # shaping K into (batch_size, num_heads, d_model//heads, seq_len)
        # for further matrix multiplication
        sqrt_d = K.constant(np.sqrt(d_model // self.num_heads),
                            dtype=K.floatx())
        q_shape = K.int_shape(q)
        k_t_shape = K.int_shape(k_transposed)
        v_shape = K.int_shape(v)
        # before performing batch_dot all tensors are being converted to 3D
        # shape (batch_size * num_heads, rows, cols) to make sure batch_dot
        # performs identically on all backends
        attention_heads = K.reshape(
            K.batch_dot(
                self.apply_dropout_if_needed(
                    K.softmax(
                        self.mask_attention_if_needed(
                            K.batch_dot(
                                K.reshape(q, (-1,) + q_shape[-2:]),
                                K.reshape(k_transposed,
                                          (-1,) + k_t_shape[-2:]))
                            / sqrt_d)),
                    training=training),
                K.reshape(v, (-1,) + v_shape[-2:])),
            (-1, self.num_heads, q_shape[-2], v_shape[-1]))
        attention_heads_merged = K.reshape(
            K.permute_dimensions(attention_heads, [0, 2, 1, 3]),
            (-1, d_model))
        attention_out = K.reshape(
            K.dot(attention_heads_merged, self.output_weights),
            (-1, out_seq_len, d_model))
        return attention_out

    def apply_dropout_if_needed(self, attention_softmax, training=None):
        if 0.0 < self.dropout < 1.0:
            def dropped_softmax():
                return K.dropout(attention_softmax, self.dropout)

            return K.in_train_phase(dropped_softmax, attention_softmax,
                                    training=training)
        return attention_softmax

    def mask_attention_if_needed(self, dot_product):
        """
        Makes sure that (when enabled) each position
        (of a decoder's self-attention) cannot attend to subsequent positions.
        This is achieved by assigning -inf (or some large negative number)
        to all invalid connections. Later softmax will turn them into zeros.
        We need this to guarantee that decoder's predictions are based
        on what has happened before the position, not after.
        The method does nothing if masking is turned off.
        :param dot_product: scaled dot-product of Q and K after reshaping them
        to 3D tensors (batch * num_heads, rows, cols)
        """
        if not self.use_masking:
            return dot_product
        last_dims = K.int_shape(dot_product)[-2:]
        low_triangle_ones = (
            np.tril(np.ones(last_dims))
            # to ensure proper broadcasting
            .reshape((1,) + last_dims))
        inverse_low_triangle = 1 - low_triangle_ones
        close_to_negative_inf = -1e9
        result = (
            K.constant(low_triangle_ones, dtype=K.floatx()) * dot_product +
            K.constant(close_to_negative_inf * inverse_low_triangle))
        return result

class MultiHeadSelfAttention(_BaseMultiHeadAttention):
    """
    Multi-head self-attention for both encoders and decoders.
    Uses only one input and has implementation which is better suited for
    such use case that more general MultiHeadAttention class.
    """

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        print("input_shape: ", input_shape, type(input_shape))
        # if not isinstance(input_shape, tuple):
        #     raise ValueError('Invalid input')
        d_model = input_shape[-1]
        self.validate_model_dimensionality(d_model)
        # These weights are concatenated matrices W_q, W_k and W_v which
        # are, in turn, concatenated W matrices of keys, queries and values
        # for each of the heads. So, essentially it's a concatenation of
        # W_q1, W_q2,..., W_qh, W_k1, W_k2,..., W_kh, W_v1, W_v2,..., W_vh
        # for all h heads.
        self.qkv_weights = self.add_weight(
            name='qkv_weights',
            shape=(d_model, d_model * 3),  # * 3 for q, k and v
            initializer='glorot_uniform',
            trainable=True)
        self.build_output_params(d_model)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        if not K.is_tensor(inputs):
            raise ValueError(
                'The layer can be called only with one tensor as an argument')
        _, seq_len, d_model = K.int_shape(inputs)
        # The first thing we need to do is to perform affine transformations
        # of the inputs to get the Queries, the Keys and the Values.
        qkv = K.dot(K.reshape(inputs, [-1, d_model]), self.qkv_weights)
        # splitting the keys, the values and the queries before further
        # processing
        pre_q, pre_k, pre_v = [
            K.reshape(
                # K.slice(qkv, (0, i * d_model), (-1, d_model)),
                qkv[:, i * d_model:(i + 1) * d_model],
                (-1, seq_len, self.num_heads, d_model // self.num_heads))
            for i in range(3)]
        attention_out = self.attention(pre_q, pre_v, pre_k, seq_len, d_model,
                                       training=kwargs.get('training'))
        return attention_out

    def compute_output_shape(self, input_shape):
        return input_shape




import keras.activations as activations
import keras.initializers as initializers


class TransformerTransition(tf.keras.layers.Layer):
    """
    Transformer transition function. The same function is used both
    in classical in Universal Transformers. Except that in Universal
    Transformer it is also shared between time steps.
    """

    def __init__(self, activation: str,
                 size_multiplier: int = 4, **kwargs):
        """
        :param activation: activation function. Must be a string or a callable.
        :param size_multiplier: How big the hidden dimension should be.
          Most of the implementation use transition functions having 4 times
          more hidden units than the model itself.
        :param kwargs: Keras-specific layer arguments.
        """
        self.activation = activations.get(activation)
        self.size_multiplier = size_multiplier
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['activation'] = activations.serialize(self.activation)
        config['size_multiplier'] = self.size_multiplier
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        d_model = input_shape[-1]
        self.weights1 = self.add_weight(
            name='weights1',
            shape=(d_model, self.size_multiplier * d_model),
            initializer='glorot_uniform',
            trainable=True)
        self.biases1 = self.add_weight(
            name='biases1',
            shape=(self.size_multiplier * d_model,),
            initializer='zeros',
            trainable=True)
        self.weights2 = self.add_weight(
            name='weights2',
            shape=(self.size_multiplier * d_model, d_model),
            initializer='glorot_uniform',
            trainable=True)
        self.biases2 = self.add_weight(
            name='biases2',
            shape=(d_model,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        d_model = input_shape[-1]
        step1 = self.activation(
            K.bias_add(
                K.dot(K.reshape(inputs, (-1, d_model)),
                      self.weights1),
                self.biases1,
                data_format='channels_last'))
        step2 = K.bias_add(
            K.dot(step1, self.weights2),
            self.biases2,
            data_format='channels_last')
        result = K.reshape(step2, (-1,) + input_shape[-2:])
        return result


class TransformerBlock:
    """
    A pseudo-layer combining together all nuts and bolts to assemble
    a complete section of both the Transformer and the Universal Transformer
    models, following description from the "Universal Transformers" paper.
    Each such block is, essentially:

    - Multi-head self-attention (masked or unmasked, with attention dropout,
      but without input dropout)
    - Residual connection,
    - Dropout
    - Layer normalization
    - Transition function
    - Residual connection
    - Dropout
    - Layer normalization

    Also check TransformerACT class if you need support for ACT (Adaptive
    Computation Time).

    IMPORTANT: The older Transformer 2017 model ("Attention is all you need")
    uses slightly different order of operations. A quote from the paper:

        "We apply dropout [33] to the output of each sub-layer,
         before it is added to the sub-layer input and normalized"

    while the Universal Transformer paper puts dropout one step *after*
    the sub-layers's output was added to its input (Figure 4 in the paper).

    In this code the order from the Universal Transformer is used, as arguably
    more reasonable. You can use classical Transformer's (2017) way of
    connecting the pieces by passing vanilla_wiring=True to the constructor.
    """
    def __init__(self, name: str, num_heads: int,
                 residual_dropout: float = 0, attention_dropout: float = 0,
                 activation:str = 'relu',
                 compression_window_size: int = None,
                 use_masking: bool = True,
                 vanilla_wiring=False):
        self.attention_layer = MultiHeadSelfAttention(
            num_heads, use_masking=use_masking, dropout=attention_dropout,
            compression_window_size=compression_window_size,
            name=f'{name}_self_attention')
        self.norm1_layer = tf.keras.layers.LayerNormalization(name=f'{name}_normalization1')
        self.dropout_layer = (
            tf.keras.layers.Dropout(residual_dropout, name=f'{name}_dropout')
            if residual_dropout > 0
            else lambda x: x)
        self.norm2_layer = tf.keras.layers.LayerNormalization(name=f'{name}_normalization2')
        self.transition_layer = TransformerTransition(
            name=f'{name}_transition', activation=activation)
        self.addition_layer = tf.keras.layers.Add(name=f'{name}_add')
        self.vanilla_wiring = vanilla_wiring

    def __call__(self, _input):
        output = self.attention_layer(_input)
        post_residual1 = (
            self.addition_layer([_input, self.dropout_layer(output)])
            if self.vanilla_wiring
            else self.dropout_layer(self.addition_layer([_input, output])))
        norm1_output = self.norm1_layer(post_residual1)
        output = self.transition_layer(norm1_output)
        post_residual2 = (
            self.addition_layer([norm1_output, self.dropout_layer(output)])
            if self.vanilla_wiring
            else self.dropout_layer(
                self.addition_layer([norm1_output, output])))
        output = self.norm2_layer(post_residual2)
        return output



class TransformerACT(tf.keras.layers.Layer):
    """
    Implements Adaptive Computation Time (ACT) for the Transformer model
    https://arxiv.org/abs/1603.08983

    How to use:

        transformer_depth = 8
        block = TransformerBlock('Transformer', num_heads=8)
        act_layer = TransformerACT()
        next_input = input  # (batch_size, sequence_length, input_size)
        for i in range(transformer_depth):
            next_input = block(next_input, step=i)
            next_input, act_weighted_output = act_layer(next_input)
        act_layer.finalize()  # adds loss
        result = act_weighted_output

    """
    def __init__(self, halt_epsilon=0.01, time_penalty=0.01, **kwargs):
        """
        :param halt_epsilon: a small constant that allows computation to halt
            after a single update (sigmoid never reaches exactly 1.0)
        :param time_penalty: parameter that weights the relative cost
            of computation versus error. The larger it is, the less
            computational steps the network will try to make and vice versa.
            The default value of 0.01 works well for Transformer.
        :param kwargs: Any standard parameters for a layer in Keras (like name)
        """
        self.halt_epsilon = halt_epsilon
        self.time_penalty = time_penalty
        self.ponder_cost = None
        self.weighted_output = None
        self.zeros_like_input = None
        self.zeros_like_halting = None
        self.ones_like_halting = None
        self.halt_budget = None
        self.remainder = None
        self.active_steps = None
        super().__init__(**kwargs)

    def get_config(self):
        return dict(
            super().get_config(),
            halt_epsilon=self.halt_epsilon,
            time_penalty=self.time_penalty)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        assert len(input_shape) == 3
        _, sequence_length, d_model = input_shape
        self.halting_kernel = self.add_weight(
            name='halting_kernel',
            shape=(d_model, 1),
            initializer='glorot_uniform',
            trainable=True)
        self.halting_biases = self.add_weight(
            name='halting_biases',
            shape=(1,),
            initializer=initializers.Constant(0.1),
            trainable=True)
        self.time_penalty_t = K.constant(self.time_penalty, dtype=K.floatx())
        return super().build(input_shape)

    def initialize_control_tensors(self, halting):
        """
        Initializes constants and some step-tracking variables
        during the first call of the layer (since for the Universal Transformer
        all the following calls are supposed to be with inputs of identical
        shapes).
        """
        self.zeros_like_halting = K.zeros_like(
            halting, name='zeros_like_halting')
        self.ones_like_halting = K.ones_like(
            halting, name='ones_like_halting')
        self.remainder = self.ones_like_halting
        self.active_steps = self.zeros_like_halting
        self.halt_budget = self.ones_like_halting - self.halt_epsilon

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        sequence_length, d_model = input_shape[-2:]
        # output of the "sigmoid halting unit" (not the probability yet)
        halting = K.sigmoid(
            K.reshape(
                K.bias_add(
                    K.dot(K.reshape(inputs, [-1, d_model]),
                          self.halting_kernel),
                    self.halting_biases,
                    data_format='channels_last'),
                [-1, sequence_length]))
        if self.zeros_like_halting is None:
            self.initialize_control_tensors(halting)
        # useful flags
        step_is_active = K.greater(self.halt_budget, 0)
        no_further_steps = K.less_equal(self.halt_budget - halting, 0)
        # halting probability is equal to
        # a. halting output if this isn't the last step (we have some budget)
        # b. to remainder if it is,
        # c. and zero for the steps that shouldn't be executed at all
        #    (out of budget for them)
        halting_prob = K.switch(
            step_is_active,
            K.switch(
                no_further_steps,
                self.remainder,
                halting),
            self.zeros_like_halting)
        self.active_steps += K.switch(
            step_is_active,
            self.ones_like_halting,
            self.zeros_like_halting)
        # We don't know which step is the last, so we keep updating
        # expression for the loss with each call of the layer
        self.ponder_cost = (
            self.time_penalty_t * K.mean(self.remainder + self.active_steps))
        # Updating "the remaining probability" and the halt budget
        self.remainder = K.switch(
            no_further_steps,
            self.remainder,
            self.remainder - halting)
        self.halt_budget -= halting  # OK to become negative

        # If none of the inputs are active at this step, then instead
        # of zeroing them out by multiplying to all-zeroes halting_prob,
        # we can simply use a constant tensor of zeroes, which means that
        # we won't even calculate the output of those steps, saving
        # some real computational time.
        if self.zeros_like_input is None:
            self.zeros_like_input = K.zeros_like(
                inputs, name='zeros_like_input')
        # just because K.any(step_is_active) doesn't work in PlaidML
        any_step_is_active = K.greater(
            K.sum(K.cast(step_is_active, 'int32')), 0)
        step_weighted_output = K.switch(
            any_step_is_active,
            K.expand_dims(halting_prob, -1) * inputs,
            self.zeros_like_input)
        if self.weighted_output is None:
            self.weighted_output = step_weighted_output
        else:
            self.weighted_output += step_weighted_output
        return [inputs, self.weighted_output]

    def compute_output_shape(self, input_shape):
        return [input_shape, input_shape]

    def finalize(self):
        self.add_loss(self.ponder_cost)


# class TransformerCoordinateEmbedding(tf.keras.layers.Layer):
#     """
#     Represents trainable positional embeddings for the Transformer model:
#
#     1. word position embeddings - one for each position in the sequence.
#     2. depth embeddings - one for each block of the model
#
#     Calling the layer with the Transformer's input will return a new input
#     with those embeddings added.
#     """
#
#     def __init__(self, max_transformer_depth: int, d_model: int=128, **kwargs):
#         self.max_depth = max_transformer_depth
#         self.d_model = d_model
#         super().__init__(**kwargs)
#
#     def get_config(self):
#         config = super().get_config()
#         config['max_transformer_depth'] = self.max_depth
#         return config
#
#     # noinspection PyAttributeOutsideInit
#     def build(self, input_shape):
#         sequence_length, d_model = input_shape[-2:]
#         print("sequence_length: ", sequence_length, d_model)
#         self.word_position_embeddings = self.add_weight(
#             # shape=(sequence_length, d_model),
#             shape=(input_shape[-2], input_shape[-1],),
#             initializer='uniform',
#             name='word_position_embeddings',
#             trainable=True)
#         print("11111")
#         self.depth_embeddings = self.add_weight(
#             shape=(self.max_depth, d_model),
#             initializer='uniform',
#             name='depth_position_embeddings',
#             trainable=True)
#         super().build(input_shape)
#
#     def call(self, inputs, **kwargs):
#         depth = kwargs.get('step')
#         if depth is None:
#             raise ValueError("Please, provide current Transformer's step"
#                              "using 'step' keyword argument.")
#         result = inputs + self.word_position_embeddings
#         if depth is not None:
#             result = result + self.depth_embeddings[depth]
#         return result


class TransformerCoordinateEmbedding(tf.keras.layers.Layer):

    def __init__(self, units=32, **kwargs):
        super(TransformerCoordinateEmbedding, self).__init__()
        self.units = units
        # self.name = name

    # def build(self, input_shape):
    #     self.w = self.add_weight(shape=(input_shape[-1], self.units),
    #                              initializer='random_normal',
    #                              trainable=True)
    #     self.b = self.add_weight(shape=(self.units,),
    #                              initializer='random_normal',
    #                              trainable=True)

    def call(self, inputs):
        return inputs #tf.matmul(inputs, self.w) + self.b

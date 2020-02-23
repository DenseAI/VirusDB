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

Building blocks with manual backward gradient computation.

Reference [The Reversible Residual Network: Backpropagation
Without Storing Activations](https://arxiv.org/pdf/1707.04585.pdf)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator

import tensorflow as tf
from tensorflow.contrib.eager.python.examples.revnet import ops


class ReversibleBlock(tf.keras.Model):
    """Single residual block contained in a _RevBlock. Each `_Residual` object has
    two _ResidualInner objects, corresponding to the `F` and `G` functions in the
    paper.
    """

    def __init__(self,
                 f_block,
                 g_block,
                 # bottleneck=False,
                 # fused=True,
                 dtype=tf.float32):
        """
        Initialization.

        Args:
          data_format: tensor data format, "NCHW"/"NHWC",
          bottleneck: use bottleneck residual if True
          fused: use fused batch normalization if True
          dtype: float16, float32, or float64
        """
        super(ReversibleBlock, self).__init__()
        self.f = f_block
        self.g = g_block

    def call(self, x, training=True):
        """Apply residual block to inputs."""
        x1, x2 = x
        f_x2 = self.f(x2, training=training)

        print("x1: ", x1)
        print("f_x2: ", f_x2)

        y1 = x1 + f_x2
        g_y1 = self.g(y1, training=training)
        y2 = x2 + g_y1

        return y1, y2

    def backward_grads(self, y, dy, training=True):
        """Manually compute backward gradients given input and output grads."""
        dy1, dy2 = dy
        y1, y2 = y

        with tf.GradientTape() as gtape:
            gtape.watch(y1)
            gy1 = self.g(y1, training=training)
        grads_combined = gtape.gradient(
            gy1, [y1] + self.g.trainable_variables, output_gradients=dy2)
        dg = grads_combined[1:]
        dx1 = dy1 + grads_combined[0]
        # This doesn't affect eager execution, but improves memory efficiency with
        # graphs
        with tf.control_dependencies(dg + [dx1]):
            x2 = y2 - gy1

        with tf.GradientTape() as ftape:
            ftape.watch(x2)
            fx2 = self.f(x2, training=training)
        grads_combined = ftape.gradient(
            fx2, [x2] + self.f.trainable_variables, output_gradients=dx1)
        df = grads_combined[1:]
        dx2 = dy2 + grads_combined[0]
        # Same behavior as above
        with tf.control_dependencies(df + [dx2]):
            x1 = y1 - fx2

        x = x1, x2
        dx = dx1, dx2
        grads = df + dg

        return x, dx, grads

    def backward_grads_with_downsample(self, x, y, dy, training=True):
        """Manually compute backward gradients given input and output grads."""
        # Splitting this from `backward_grads` for better readability
        x1, x2 = x
        y1, _ = y
        dy1, dy2 = dy

        with tf.GradientTape() as gtape:
            gtape.watch(y1)
            gy1 = self.g(y1, training=training)
        grads_combined = gtape.gradient(
            gy1, [y1] + self.g.trainable_variables, output_gradients=dy2)
        dg = grads_combined[1:]
        dz1 = dy1 + grads_combined[0]

        # dx1 need one more step to backprop through downsample
        with tf.GradientTape() as x1tape:
            x1tape.watch(x1)
            z1 = x1 #ops.downsample(x1, self.filters // 2, self.strides, axis=self.axis)
        dx1 = x1tape.gradient(z1, x1, output_gradients=dz1)

        with tf.GradientTape() as ftape:
            ftape.watch(x2)
            fx2 = self.f(x2, training=training)
        grads_combined = ftape.gradient(
            fx2, [x2] + self.f.trainable_variables, output_gradients=dz1)
        dx2, df = grads_combined[0], grads_combined[1:]

        # dx2 need one more step to backprop through downsample
        with tf.GradientTape() as x2tape:
            x2tape.watch(x2)
            z2 = x2 #ops.downsample(x2, self.filters // 2, self.strides, axis=self.axis)
        dx2 += x2tape.gradient(z2, x2, output_gradients=dy2)

        dx = dx1, dx2
        grads = df + dg

        return dx, grads


class ReversibleSequence(tf.keras.Model):
    """
    Single reversible block containing several `Reversible` blocks.
    Each `Reversible` block in turn contains two _ResidualInner blocks,
    corresponding to the `F`/`G` functions in the paper.
    """
    def __init__(self,
                 f_block,
                 g_block,
                 # n_res,
                 # input_shape,
                 # batch_norm_first=False,
                 # data_format="channels_first",
                 # bottleneck=False,
                 # fused=True,
                 dtype=tf.float32):
        """
        Initialization.

        Args:
          n_res: number of residual blocks
          input_shape: length 3 list/tuple of integers
          data_format: tensor data format, "NCHW"/"NHWC"
          bottleneck: use bottleneck residual if True
          fused: use fused batch normalization if True
          dtype: float16, float32, or float64
        """
        super(ReversibleSequence, self).__init__()
        self.blocks = tf.contrib.checkpoint.List()
        for i in range(1):
            # curr_batch_norm_first = batch_norm_first and i == 0
            # curr_strides = strides if i == 0 else (1, 1)
            block = ReversibleBlock(
                f_block,
                g_block,
                # filters,
                # curr_strides,
                # input_shape,
                # batch_norm_first=curr_batch_norm_first,
                # data_format=data_format,
                # bottleneck=bottleneck,
                # fused=fused,
                dtype=dtype)
            self.blocks.append(block)

    def call(self, h, training=True):
        """Apply reversible block to inputs."""

        for block in self.blocks:
            h = block(h, training=training)
        return h

    def backward_grads(self, x, y, dy, training=True):
        """Apply reversible block backward to outputs."""

        grads_all = []
        for i in reversed(range(len(self.blocks))):
            block = self.blocks[i]
            # if i == 0:
            #     # First block usually contains downsampling that can't be reversed
            #     dy, grads = block.backward_grads_with_downsample(
            #         x, y, dy, training=True)
            # else:
            y, dy, grads = block.backward_grads(y, dy, training=training)
            grads_all = grads + grads_all

        return dy, grads_all

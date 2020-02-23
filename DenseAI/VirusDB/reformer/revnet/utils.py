import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import initializers

class ScaleNorm(layers.Layer):
    def __init__(self, emb, eps, g_initializer='zeros'):
        super(ScaleNorm, self).__init__()
        self.g_initializer = initializers.get(g_initializer)
        self.eps = eps

    def build(self, input_shape):
        self.g = self.add_weight(
            'g',
            shape=[1, ],
            initializer=self.g_initializer,
            dtype=self.dtype,
            trainable=True)

    def call(self, inputs):
        x = inputs
        n = tf.norm(inputs, axis=-1, keepdims=True).clip_by_value(min=self.eps)
        return x / n * self.g


class WithNorm(layers.Layer):
    def __init__(self, norm_class, emb, fn):
        super(WithNorm, self).__init__()
        self.emb = emb
        if isinstance(norm_class, ScaleNorm):
            self.norm = norm_class(emb)
        else:
            self.norm = norm_class()

        self.fn = fn

    def call(self, inputs):
        inputs = self.norm(inputs)
        return self.fn(inputs)

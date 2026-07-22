"""Custom TensorFlow/Keras layers used by the notebook's Transformer model."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="PeerFeedback")
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.projection_dim = self.embed_dim // self.num_heads
        self.query_dense = layers.Dense(self.embed_dim)
        self.key_dense = layers.Dense(self.embed_dim)
        self.value_dense = layers.Dense(self.embed_dim)
        self.combine_heads = layers.Dense(self.embed_dim)

    @staticmethod
    def attention(query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        return tf.matmul(weights, value)

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.separate_heads(self.query_dense(inputs), batch_size)
        key = self.separate_heads(self.key_dense(inputs), batch_size)
        value = self.separate_heads(self.value_dense(inputs), batch_size)
        attention = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        return self.combine_heads(concat_attention)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config


@keras.utils.register_keras_serializable(package="PeerFeedback")
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.rate = float(rate)
        self.att = MultiHeadSelfAttention(self.embed_dim, self.num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(self.ff_dim, activation="relu"), layers.Dense(self.embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="PeerFeedback")
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = int(maxlen)
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.token_emb = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)
        self.pos_emb = layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        positions = self.pos_emb(positions)
        return self.token_emb(x) + positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "maxlen": self.maxlen,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config

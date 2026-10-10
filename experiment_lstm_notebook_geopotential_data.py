"""Reusable experiment functions extracted from LSTM Notebook (Geopotential Data).ipynb."""

from __future__ import annotations

from tensorflow import keras
import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def create_encoder(
    latent_dim,
    batch_size=32,
    input_shape=(32, 64, 1),
    number_of_filters=(32, 64, 128),
    k_shape=(3, 3, 3),
    stride=(2, 2, 2),
    LSTM_units=64 * 64,
):
    from tensorflow import keras
    import tensorflow as tf

    encoder_inputs = tf.keras.Input(shape=input_shape, batch_size=batch_size)
    x = encoder_inputs
    for i in range(len(number_of_filters)):
        x = tf.keras.layers.Conv2D(
            filters=number_of_filters[i],
            kernel_size=k_shape[i],
            activation="relu",
            strides=(stride[i], stride[i]),
            padding="same",
        )(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_dim + latent_dim)(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def create_decoder(
    latent_dim,
    batch_size=32,
    input_shape=(32, 64, 1),
    number_of_filters=(32, 64, 128),
    k_shape=(3, 3, 3),
    stride=(2, 2, 2),
):
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow as tf

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(
        input_shape[0]
        // 2 ** len(number_of_filters)
        * (input_shape[1] // 2 ** len(number_of_filters))
        * number_of_filters[-1],
        activation="relu",
    )(latent_inputs)
    x = tf.keras.layers.Reshape(
        (
            input_shape[0] // 2 ** len(number_of_filters),
            input_shape[1] // 2 ** len(number_of_filters),
            number_of_filters[-1],
        )
    )(x)
    for i in range(len(number_of_filters)):
        j = len(number_of_filters) - 1 - i
        x = tf.keras.layers.Conv2DTranspose(
            filters=number_of_filters[j],
            kernel_size=k_shape[j],
            activation="relu",
            strides=(stride[j], stride[j]),
            padding="same",
        )(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        data, target = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(target, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def normalize_data(data):
    v_min, v_max = (data.min(), data.max())
    if v_max == v_min:
        return data * 0.0
    return (data - v_min) / (v_max - v_min)


def expand_data(data):
    import numpy as np

    return np.expand_dims(data, axis=-1)


def create_shifted_frames(data):
    x = data[0 : data.shape[0] - 1, :, :, :]
    y = data[1 : data.shape[0], :, :, :]
    return (x, y)

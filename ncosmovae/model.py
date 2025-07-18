import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import MeanSquaredError

class NcosmoVAE(Model):
    def __init__(self, image_size=256, latent_dim=512, **kwargs):
        super(NcosmoVAE, self).__init__(**kwargs)
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.mse_loss = MeanSquaredError()

    def build_encoder(self):
        encoder_input = layers.Input(shape=(self.image_size, self.image_size, 1))
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_input)
        x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(512, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(100, activation="relu")(x)
        x = layers.Dense(100, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)
        z = layers.Lambda(self.sampling)([z_mean, z_log_var])
        return Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

    def build_decoder(self):
        latent_input = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(100, activation="relu")(latent_input)
        x = layers.Dense(100, activation="relu")(x)
        x = layers.Dense(32 * 32 * self.image_size, activation="relu")(x)
        x = layers.Reshape((32, 32, self.image_size))(x)
        x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        output = layers.Conv2DTranspose(1, 3, activation="linear", padding="same")(x)
        return Model(latent_input, output, name="decoder")

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def train_step(self, data):
        if isinstance(data, tuple):
            x, y = data
        else:
            x = y = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(self.mse_loss(y, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            x, y = data
        else:
            x = y = data

        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(self.mse_loss(y, reconstruction))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }

    def call(self, inputs):
        _, _, z = s


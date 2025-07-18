import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from ncosmovae import NcosmoVAE, load_dataset

# -----------------------------
# CONFIG USING OPTIMIZED HYPERPARAMETERS FOR batch_size, kernel_size, dense_units
# -----------------------------
image_size = 256
latent_dim = 512
kernel_size = 5
dense_units = 256
batch_size = 4
epochs = 50

proj_path = "data/Projections_axis_off"
halo_path = "data/HALOS_Axis_off/Axis_off"

# -----------------------------
# LOAD DATA
# -----------------------------
X, Y = load_dataset(proj_path, halo_path, image_size=image_size)
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

# -----------------------------
# MODEL + EARLY STOPPING
# -----------------------------
vae = NcosmoVAE(
    image_size=image_size,
    latent_dim=latent_dim,
    kernel_size=kernel_size,
    dense_units=dense_units
)
vae.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=1e-5,
    restore_best_weights=True
)

history = vae.fit(
    x=X_train,
    y=Y_train,
    validation_data=(X_val, Y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop]
)

# -----------------------------
# SAVE ENCODER & DECODER
# -----------------------------
vae.encoder.save("saved_models/encoder.keras")
vae.decoder.save("saved_models/decoder.keras")
print("Models saved to 'saved_models/'")

# -----------------------------
# GENERATE FROM LATENT SPACE
# -----------------------------
z_random = tf.random.normal((1, latent_dim))
generated = vae.decoder.predict(z_random)

plt.imshow(generated[0, :, :, 0], cmap="gray")
plt.title("Generated Image from Latent Vector")
plt.axis("off")
plt.tight_layout()
plt.show()

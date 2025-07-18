import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ncosmovae import NcosmoVAE, load_dataset
from tensorflow.keras.callbacks import EarlyStopping


# Config
image_size = 256
latent_dim = 512
proj_path = "data/Projections_axis_off"
halo_path = "data/HALOS_Axis_off/Axis_off"

# Load data
X, Y = load_dataset(proj_path, halo_path, image_size=image_size)
print(f"Loaded data: X={X.shape}, Y={Y.shape}")

# Split
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

# Instantiate and compile
vae = NcosmoVAE(image_size=image_size, latent_dim=latent_dim)
vae.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=1e-5,
    restore_best_weights=True
)

history = vae.fit(
    x=X_train,
    y=Y_train,
    validation_data=(X_val, Y_val),
    epochs=20,
    batch_size=4,
    callbacks=[early_stop]
)

# Plot loss
plt.plot(history.history['loss'], label="Total Loss")
plt.plot(history.history['reconstruction_loss'], label="Reconstruction")
plt.plot(history.history['kl_loss'], label="KL")
plt.legend()
plt.title("Training Loss")
plt.show()

# Generate samples
z = tf.random.normal((8, latent_dim))
generated = vae.decoder.predict(z)

for i in range(len(generated)):
    plt.imshow(generated[i, :, :, 0], cmap="gray")
    plt.axis("off")
    plt.title(f"Generated {i+1}")
    plt.show()


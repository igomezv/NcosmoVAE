import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from ncosmovae import NcosmoVAE, load_dataset

# -----------------------------
# CONFIGURATION
# -----------------------------
image_size = 256
latent_dim = 512
kernel_size = 5
dense_units = 256

# Paths for trained models
encoder_path = "saved_models/encoder.keras"
decoder_path = "saved_models/decoder.keras"

# NEW DATA PATH
proj_path = "data/other_projections"

# Output directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("latent_representations", exist_ok=True)

# -----------------------------
# LOAD NEW DATA
# -----------------------------
X_new, _ = load_dataset(proj_path, proj_path, image_size=image_size)
print(f"Loaded new dataset: {X_new.shape[0]} samples")

# -----------------------------
# LOAD TRAINED MODELS
# -----------------------------
# Try loading full models (if saved with .keras)
try:
    from ncosmovae import NcosmoVAE

    # Create temporary instance to access sampling if needed
    vae_temp = NcosmoVAE(image_size=image_size, latent_dim=latent_dim, kernel_size=kernel_size, dense_units=dense_units)
    encoder = load_model(encoder_path, custom_objects={"sampling": vae_temp.sampling})
    decoder = load_model(decoder_path)
    print("Loaded full encoder/decoder models from .keras files.")

except Exception as e:
    print("Could not load full models, trying to load weights only...")
    vae = NcosmoVAE(image_size=image_size, latent_dim=latent_dim, kernel_size=kernel_size, dense_units=dense_units)
    vae.encoder.load_weights(encoder_path)
    vae.decoder.load_weights(decoder_path)
    encoder, decoder = vae.encoder, vae.decoder
    print("Loaded encoder/decoder weights successfully.")

# -----------------------------
# ENCODE NEW DATA (LATENT REPRESENTATIONS)
# -----------------------------
print("Encoding new data into latent space...")
z_mean, z_log_var, z = encoder.predict(X_new, batch_size=4)
latent_reps = z  # only the latent vector
print(f"Latent representations shape: {latent_reps.shape}")

# Save latent vectors
np.save("latent_representations/latent_vectors.npy", latent_reps)
np.save("latent_representations/z_mean.npy", z_mean)
np.save("latent_representations/z_log_var.npy", z_log_var)
print("Saved latent representations to 'latent_representations/'")

# -----------------------------
# DECODE (RECONSTRUCT) IMAGES
# -----------------------------
print("Decoding latent representations (reconstructing images)...")
reconstructed = decoder.predict(latent_reps, batch_size=4)
print(f"Reconstructed images shape: {reconstructed.shape}")

# Save outputs
np.save("outputs/reconstructed_images.npy", reconstructed)
print("Saved reconstructed images to 'outputs/reconstructed_images.npy'")

# -----------------------------
# VISUALIZE A SAMPLE
# -----------------------------
idx = 0
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(X_new[idx, :, :, 0], cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed[idx, :, :, 0], cmap="gray")
plt.title("Reconstructed Image")
plt.axis("off")

plt.tight_layout()
plt.show()


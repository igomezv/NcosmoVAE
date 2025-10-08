import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
latent_dim = 512
num_samples = 5
decoder_path = "saved_models/decoder.keras"

# Output directory
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# LOAD DECODER
# -----------------------------
decoder = load_model(decoder_path)
print("Decoder model loaded successfully.")

# -----------------------------
# GENERATE RANDOM LATENT VECTORS
# -----------------------------
z_random = tf.random.normal((num_samples, latent_dim))
generated_images = decoder.predict(z_random)

print(f"Generated {num_samples} images from random latent vectors.")
print(f"Generated images shape: {generated_images.shape}")

# -----------------------------
# SAVE INDIVIDUAL IMAGES
# -----------------------------
for i in range(num_samples):
    img = generated_images[i, :, :, 0]
    plt.imsave(f"outputs/generated_{i+1}.png", img, cmap="gray")

print(f"Saved generated images to 'outputs/generated_#.png'")

# -----------------------------
# OPTIONAL: DISPLAY GRID OF IMAGES
# -----------------------------
fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(generated_images[i, :, :, 0], cmap="gray")
    ax.set_title(f"Sample {i+1}")
    ax.axis("off")

plt.tight_layout()
plt.show()


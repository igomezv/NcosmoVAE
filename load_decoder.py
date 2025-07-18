from tensorflow.keras.models import load_model

# Load the decoder
decoder = load_model("saved_models/decoder.keras")

# Generate image from a random latent vector
import tensorflow as tf
import matplotlib.pyplot as plt

latent_dim = 512
z = tf.random.normal((1, latent_dim))
generated = decoder.predict(z)

# Plot the result
plt.imshow(generated[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.title("Generated from Loaded Decoder")
plt.show()


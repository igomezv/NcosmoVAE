import os
import numpy as np
import tensorflow as tf
from skimage import io
import cv2

def preprocess(image_path, image_size=256):
    image = io.imread(image_path)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    image = image / 255.0
    image = tf.image.rot90(image, k=3).numpy()
    if image.ndim == 3 and image.shape[2] >= 2:
        return image[:, :, 1]  # green channel
    else:
        return image

def load_dataset(proj_dir, halo_dir, image_size=256, noise_level=0.1):
    proj_files = sorted([os.path.join(proj_dir, f) for f in os.listdir(proj_dir) if f.endswith(".png")])
    halo_files = sorted([os.path.join(halo_dir, f) for f in os.listdir(halo_dir) if f.endswith(".png")])
    assert len(proj_files) == len(halo_files), "Mismatch in projection and halo images"

    idx = np.random.permutation(len(proj_files))
    proj_files = [proj_files[i] for i in idx]
    halo_files = [halo_files[i] for i in idx]

    X = np.array([preprocess(p, image_size) for p in proj_files])
    Y = np.array([preprocess(p, image_size) for p in halo_files])

    # Add noise
    noisy_X = X + noise_level * np.random.rand(*X.shape)
    noisy_Y = Y + noise_level * np.random.rand(*Y.shape)

    X = np.concatenate([X, noisy_X], axis=0)[..., np.newaxis]
    Y = np.concatenate([Y, noisy_Y], axis=0)[..., np.newaxis]

    return X.astype(np.float32), Y.astype(np.float32)


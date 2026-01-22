import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from ncosmovae import load_dataset, NcosmoVAE

# -----------------------------
# CONFIG
# -----------------------------
image_size = 256
latent_dim = 512
batch_size = 32
encoder_path = "saved_models/encoder.keras"

proj_path = "data/Projections_axis_off"
halo_path = "data/HALOS_Axis_off/Axis_off"

# Output directory
os.makedirs("latent_analysis", exist_ok=True)

# -----------------------------
# LOAD DATA AND SPLIT
# -----------------------------
print("Loading dataset...")
X, Y = load_dataset(proj_path, halo_path, image_size=image_size)

# Split dataset (use same split as training)
split = int(0.0 * len(X))
X_test = X[split:]

print(f"Test set size: {len(X_test)} samples")

# -----------------------------
# LOAD ENCODER
# -----------------------------
print("Loading encoder model...")
try:
    # Try to load the saved model with safe mode disabled
    encoder = load_model(encoder_path, safe_mode=False)
    print("Encoder model loaded successfully from saved file.")
except Exception as e:
    print(f"Failed to load saved encoder: {e}")
    print("Creating encoder from scratch with same configuration...")
    
    # Create VAE with same configuration and extract encoder
    vae = NcosmoVAE(
        image_size=image_size,
        latent_dim=latent_dim,
        kernel_size=5,
        dense_units=256
    )
    
    # Try to load weights if they exist in .h5 format
    if os.path.exists(encoder_path.replace('.keras', '.h5')):
        try:
            vae.load_weights(encoder_path.replace('.keras', '.h5'))
            print("Loaded weights from .h5 file")
        except:
            print("Warning: Could not load weights, using random initialization")
    
    encoder = vae.encoder
    print("Encoder created from VAE model")

# -----------------------------
# ENCODE TEST DATA
# -----------------------------
print("Encoding test data...")

# Use full dataset size
batch_size = len(X_test)

X_sample = X_test[:batch_size]

print(f"Processing {len(X_sample)} samples")

# Encode the samples
z_mean_X, z_log_var_X, z_X = encoder.predict(X_sample, verbose=1)

print(f"Encoded data shapes:")
print(f"  Projections: {z_mean_X.shape}")

# Create labels for visualization
labels_X = np.array(['Projection'] * len(z_mean_X))

# -----------------------------
# 2D VISUALIZATIONS (PCA & T-SNE)
# -----------------------------
print("Generating 2D projections...")

# Standardize data for better results
scaler = StandardScaler()
z_scaled = scaler.fit_transform(z_mean_X)

# PCA Analysis
pca = PCA(n_components=2)
z_pca = pca.fit_transform(z_scaled)

# t-SNE Analysis
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
z_tsne = tsne.fit_transform(z_scaled)

# Plot PCA separately
plt.figure(figsize=(8, 6))
plt.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.7, label='Projection', s=50)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA of Latent Space')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('latent_analysis/pca_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot t-SNE separately
plt.figure(figsize=(8, 6))
plt.scatter(z_tsne[:, 0], z_tsne[:, 1], alpha=0.7, label='Projection', s=50)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Latent Space')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('latent_analysis/tsne_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# LATENT DIMENSION DISTRIBUTIONS
# -----------------------------
print("Analyzing latent dimension distributions...")

# Sample first 20 dimensions for visualization
n_dims_to_plot = min(20, latent_dim)
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
axes = axes.flatten()

for i in range(n_dims_to_plot):
    ax = axes[i]
    
    # Plot distributions for projections only
    ax.hist(z_mean_X[:, i], bins=15, alpha=0.6, label='Projection', density=True)
    
    ax.set_title(f'Dim {i+1}', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    if i == 0:
        ax.legend(fontsize=8)

# Remove empty subplots
for i in range(n_dims_to_plot, len(axes)):
    fig.delaxes(axes[i])

plt.suptitle('Latent Dimension Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('latent_analysis/latent_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# CORRELATION MATRIX
# -----------------------------
print("Computing correlation matrix...")

# Sample first 50 dimensions for correlation analysis
n_dims_corr = min(50, latent_dim)
z_sample = z_mean_X[:, :n_dims_corr]

correlation_matrix = np.corrcoef(z_sample.T)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
            square=True, cbar_kws={'shrink': 0.8})
plt.title(f'Latent Space Correlation Matrix (First {n_dims_corr} dimensions)')
plt.tight_layout()
plt.savefig('latent_analysis/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# STATISTICAL ANALYSIS
# -----------------------------
print("Computing statistical analysis...")

# Compute statistics for projections
stats_X = {
    'mean': np.mean(z_mean_X, axis=0),
    'std': np.std(z_mean_X, axis=0),
    'min': np.min(z_mean_X, axis=0),
    'max': np.max(z_mean_X, axis=0)
}

# Plot statistical comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Mean comparison
axes[0, 0].plot(stats_X['mean'], label='Projection', alpha=0.7)
axes[0, 0].set_title('Mean Values per Latent Dimension')
axes[0, 0].set_xlabel('Latent Dimension')
axes[0, 0].set_ylabel('Mean Value')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Standard deviation comparison
axes[0, 1].plot(stats_X['std'], label='Projection', alpha=0.7)
axes[0, 1].set_title('Standard Deviation per Latent Dimension')
axes[0, 1].set_xlabel('Latent Dimension')
axes[0, 1].set_ylabel('Standard Deviation')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# KL Divergence analysis (average per dimension)
kl_divergence = -0.5 * (1 + z_log_var_X - np.square(z_mean_X) - np.exp(z_log_var_X))
kl_mean = np.mean(kl_divergence, axis=0)

axes[1, 0].plot(kl_mean)
axes[1, 0].set_title('Average KL Divergence per Latent Dimension')
axes[1, 0].set_xlabel('Latent Dimension')
axes[1, 0].set_ylabel('KL Divergence')
axes[1, 0].grid(True, alpha=0.3)

# Distribution of overall KL divergence
kl_total = np.sum(kl_divergence, axis=1)
axes[1, 1].hist(kl_total, bins=30, alpha=0.7, density=True)
axes[1, 1].set_title('Distribution of Total KL Divergence')
axes[1, 1].set_xlabel('Total KL Divergence')
axes[1, 1].set_ylabel('Density')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('latent_analysis/statistical_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# SAVE SUMMARY REPORT
# -----------------------------
print("Generating summary report...")

summary_text = f"""
=== Latent Space Analysis Summary ===

Dataset Information:
- Test set size: {len(X_test)} samples
- Projections: {len(z_mean_X)} samples
- Latent dimensions: {latent_dim}

PCA Analysis:
- Explained variance (PC1): {pca.explained_variance_ratio_[0]:.2%}
- Explained variance (PC2): {pca.explained_variance_ratio_[1]:.2%}
- Total explained variance (PC1+PC2): {sum(pca.explained_variance_ratio_[:2]):.2%}

Statistical Summary:
- Projection latent mean: {np.mean(stats_X['mean']):.4f} ± {np.std(stats_X['mean']):.4f}
- Average KL divergence: {np.mean(kl_total):.4f}

Correlation Analysis:
- Mean absolute correlation: {np.mean(np.abs(correlation_matrix - np.eye(n_dims_corr))):.4f}
- Max correlation: {np.max(correlation_matrix - np.eye(n_dims_corr)):.4f}

Generated Files:
- pca_visualization.png
- tsne_visualization.png
- latent_distributions.png
- correlation_matrix.png
- statistical_analysis.png
"""

with open('latent_analysis/summary_report.txt', 'w') as f:
    f.write(summary_text)

print("Latent space analysis complete!")
print("Results saved to 'latent_analysis/' directory")
print(summary_text)

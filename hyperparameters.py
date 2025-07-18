import optuna
from optuna.samplers import NSGAIISampler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


from ncosmovae import NcosmoVAE, load_dataset

# -----------------------
# CONSTANTS
# -----------------------
IMAGE_SIZE = 256
LATENT_DIM = 512
N_EPOCHS = 20

PROJ_PATH = "data/Projections_axis_off"
HALO_PATH = "data/HALOS_Axis_off/Axis_off"

# -----------------------
# LOAD DATA ONCE
# -----------------------
X, Y = load_dataset(PROJ_PATH, HALO_PATH, image_size=IMAGE_SIZE)
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

# -----------------------
# OPTUNA OBJECTIVE
# -----------------------
def objective(trial):
    # Hyperparameter search space
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
    dense_units = trial.suggest_int("dense_units", 64, 256, step=64)

    # Build model
    vae = NcosmoVAE(
        image_size=IMAGE_SIZE,
        latent_dim=LATENT_DIM,
        kernel_size=kernel_size,
        dense_units=dense_units
    )
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=1e-5,
    restore_best_weights=True)


    # Train
    history = vae.fit(
        x=X_train, y=Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=N_EPOCHS,
        callbacks=[early_stop],
        verbose=0
    )

    final_val_rec = history.history["val_reconstruction_loss"][-1]
    final_val_kl = history.history["val_kl_loss"][-1]

    # NSGAII minimizes both objectives
    return final_val_rec, final_val_kl

# -----------------------
# RUN OPTUNA STUDY
# -----------------------
study = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=NSGAIISampler()
)

study.optimize(objective, n_trials=10)

# -----------------------
# REPORT RESULTS
# -----------------------
print("\n✅ Pareto-optimal trials:")
for i, trial in enumerate(study.best_trials):
    print(f"Trial {i+1}:")
    print(f"  Rec. Loss = {trial.values[0]:.5f}, KL Loss = {trial.values[1]:.5f}")
    print(f"  Params    = {trial.params}")

# -----------------------
# PLOT PARETO FRONT
# -----------------------
recs = [t.values[0] for t in study.best_trials]
kls = [t.values[1] for t in study.best_trials]

plt.figure(figsize=(8,6))
plt.scatter(recs, kls, c="red", s=50)
plt.xlabel("Reconstruction Loss")
plt.ylabel("KL Divergence")
plt.title("Pareto Front (Optuna NSGAII)")
plt.grid(True)
plt.tight_layout()
plt.show()


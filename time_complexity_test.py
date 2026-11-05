import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

from koopman.backbone import *
from koopman.train_test import set_seed, train
from koopman.koopman import *
from koopman.train_test import *

# ==== Lorenz System Simulator ====
def simulate_lorenz(sigma=10.0, rho=28.0, beta=8/3, dt=0.01, T=20.0):
    num_steps = int(T / dt)
    x = torch.zeros((num_steps, 3))
    x[0, 0] = 1.0
    x[0, 1] = 1.0
    x[0, 2] = 1.0
    for t in range(num_steps - 1):
        x_t = x[t]
        dx = sigma * (x_t[1] - x_t[0])
        dy = x_t[0] * (rho - x_t[2]) - x_t[1]
        dz = x_t[0] * x_t[1] - beta * x_t[2]

        x[t + 1, 0] = x_t[0] + dt * dx + 0.1 * torch.randn(1)
        x[t + 1, 1] = x_t[1] + dt * dy + 0.1 * torch.randn(1)
        x[t + 1, 2] = x_t[2] + dt * dz + 0.1 * torch.randn(1)
    return x

# ==== Rössler 4-State System Simulator ====
def simulate_rossler_system(a=0.2, b=0.2, c=5.7, gamma=0.1, delta=0.2, dt=0.01, T=20.0):
    num_steps = int(T / dt)
    x = torch.zeros((num_steps, 4))
    x[0] = torch.tensor([1.0, 1.0, 1.0, 0.5])

    for t in range(num_steps - 1):
        x1, x2, x3, x4 = x[t]
        dx1 = -x2 - x3
        dx2 = x1 + a * x2
        dx3 = b + x3 * (x1 - c)
        dx4 = -gamma * x4 + delta * x3

        x[t + 1, 0] = x1 + dt * dx1 + 0.1 * torch.randn(1)
        x[t + 1, 1] = x2 + dt * dx2 + 0.1 * torch.randn(1)
        x[t + 1, 2] = x3 + dt * dx3 + 0.01 * torch.randn(1)
        x[t + 1, 3] = x4 + dt * dx4 + 0.01 * torch.randn(1)
    return x

# ==== Unbounded Random ODE Simulator ====
def simulate_random_ode_unbounded(
    num_vars=3,
    dt=0.01,
    T=20.0,
    noise_std=0.1,
    seed=None,
):
    """
    Simulates a randomly generated nonlinear ODE system.

    Each variable follows:
        dx_i/dt = linear terms
                  + quadratic terms
                  + sinusoidal terms
                  + optional cross-variable interactions
    """

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    num_steps = int(T / dt)

    # Initial condition
    x = torch.zeros((num_steps, num_vars))
    x[0] = torch.randn(num_vars)

    # Random ODE parameters
    linear = torch.randn(num_vars, num_vars) * 1.0
    quadratic = torch.randn(num_vars, num_vars) * 0.2
    sine_coeff = torch.randn(num_vars, num_vars) * 0.5
    bias = torch.randn(num_vars) * 0.1

    for t in range(num_steps - 1):
        x_t = x[t]

        # Linear dynamics: A @ x
        dx_linear = linear @ x_t

        # Nonlinear quadratic dynamics
        dx_quad = torch.zeros(num_vars)
        for i in range(num_vars):
            for j in range(num_vars):
                dx_quad[i] += quadratic[i, j] * x_t[j] ** 2

        # Sinusoidal interactions
        dx_sine = sine_coeff @ torch.sin(x_t)

        # Complete derivative
        dx = (
            dx_linear
            + dx_quad
            + dx_sine
            + bias
        )

        # Euler integration + noise
        x[t + 1] = (
            x_t
            + dt * dx
            + noise_std * torch.randn(num_vars)
        )

    return x

# ==== Bounded Random ODE Simulator ====
def simulate_random_ode_bounded(
    num_vars=3,
    dt=0.01,
    T=20.0,
    noise_std=0.01,
    seed=None,
    state_scale=5.0,
):
    """
    Simulates a bounded random nonlinear ODE system.

    The state is kept within approximately [-state_scale, state_scale].
    """

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    num_steps = int(T / dt)

    # State
    x = torch.zeros((num_steps, num_vars))

    # Bounded initial condition
    x[0] = torch.randn(num_vars) * 0.5

    # Random parameters
    # Keep the linear dynamics relatively weak/stable.
    linear = torch.randn(num_vars, num_vars) * 0.3
    linear -= 0.8 * torch.eye(num_vars)

    nonlinear = torch.randn(num_vars, num_vars) * 0.5
    sine_coeff = torch.randn(num_vars, num_vars) * 0.3
    bias = torch.randn(num_vars) * 0.1

    for t in range(num_steps - 1):
        x_t = x[t]

        # Normalize state before nonlinear interactions
        z = x_t / state_scale

        # Linear dynamics
        dx_linear = linear @ z

        # Bounded nonlinear interactions
        dx_nonlinear = nonlinear @ torch.tanh(z)

        # Bounded sinusoidal interactions
        dx_sine = sine_coeff @ torch.sin(z)

        # Small bias
        dx = (
            dx_linear
            + dx_nonlinear
            + dx_sine
            + bias
        )

        # Euler integration
        x_next = x_t + dt * dx

        # Soft bounding:
        # maps arbitrarily large values smoothly back into
        # approximately [-state_scale, state_scale].
        x_next = state_scale * torch.tanh(x_next / state_scale)

        # Small observation/process noise
        x_next += noise_std * torch.randn(num_vars)

        # Hard safety bound
        x[t + 1] = torch.clamp(
            x_next,
            -state_scale,
            state_scale,
        )

    return x

# ==== Koopformer PRO Wrappers ====
class PatchTSTWrapper(nn.Module):
    def __init__(self, seq_len, patch_len, horizon):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim= 3, seq_len=seq_len, patch_len=patch_len,d_model= 16, num_heads=2, dim_ff=64, num_layers= 2)
        self.koopman = StrictStableKoopmanOperator(16)
        self.fc = nn.Linear(16, horizon * 3)

    def forward(self, x):
        #x = x.reshape(x.shape[0], 1, -1)
        latent = self.backbone(x)
        latent_next, K, eigvals = self.koopman(latent)
        out = self.fc(latent_next)
        return out, latent, latent_next, K, eigvals

class AutoformerWrapper(nn.Module):
    def __init__(self, seq_len, patch_len, horizon):
        super().__init__()
        self.backbone = SimpleAutoformer(input_dim= 3, horizon=16, input_len=seq_len, patch_len=patch_len,d_model= 16, num_heads=2, dim_ff=64, num_layers= 2)
        #self.to_latent = nn.Linear(horizon, 16)
        self.koopman = StrictStableKoopmanOperator(16)
        self.fc = nn.Linear(16, horizon * 3)

    def forward(self, x):
        #x = x.reshape(x.shape[0], -1, 1)
        latent = self.backbone(x)
        latent_next, K, eigvals = self.koopman(latent)
        out = self.fc(latent_next)
        return out, latent, latent_next, K, eigvals

class InformerWrapper(nn.Module):
    def __init__(self, seq_len, patch_len, horizon):
        super().__init__()
        self.backbone = InformerSparse(input_dim= 3, seq_len=seq_len, patch_len=patch_len,d_model= 16, num_heads=2, dim_ff=64, num_layers= 2)
        self.koopman = StrictStableKoopmanOperator(16)
        self.fc = nn.Linear(16, horizon * 3)

    def forward(self, x):
        latent = self.backbone(x)
        latent_next, K, eigvals = self.koopman(latent)
        out = self.fc(latent_next)
        return out, latent, latent_next, K, eigvals

# ==== TRAINING FUNCTION ====
def train_and_eval(model, x_in, x_out, epochs=3000):
    opt = optim.Adam(model.parameters(), lr=0.001)
    mse = nn.MSELoss()
    eig_hist = []

    for e in range(epochs):
        model.train()
        opt.zero_grad()
        preds, latent, latent_next, _, eigvals = model(x_in)
        loss = mse(preds, x_out) + 0.1 * torch.relu((latent_next.norm(dim=1) ** 2 - latent.norm(dim=1) ** 2)).mean()
        loss.backward()
        opt.step()

        if e % 20 == 0:
            eig_hist.append(eigvals.detach().cpu().numpy())

    model.eval()
    with torch.no_grad():
        preds, _, _, _, _ = model(x_in)
    return preds.cpu().numpy(), x_out.cpu().numpy(), np.array(eig_hist)

# ==== MAIN ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.version.cuda)

data = simulate_lorenz(10)

'''
data = simulate_random_ode_bounded(
    num_vars=3,
    dt=0.01,
    T=20.0,
    noise_std=0.1,
    seed=42,
)'''

seq_len = 100
patch_len = 10
horizon = 5
epochs = 2000
d_model = 64

x_in, x_out = [], []

for i in range(len(data) - seq_len - horizon + 1):
    x_in.append(data[i:i + seq_len])
    x_out.append(
        data[i + seq_len:i + seq_len + horizon].reshape(-1)
    )

x_in = torch.stack(x_in).to(device)
x_out = torch.stack(x_out).to(device)

print("x_in shape:", x_in.shape)
print("x_out shape:", x_out.shape)

input_dim = 3

models = {
    "LSTM": SimpleLSTMForecaster(input_dim, 64, 2, horizon).to(device),
    "PatchTST": Transformer_PatchTST(input_dim, seq_len, horizon, patch_len=patch_len, d_model=d_model).to(device),
    "Autoformer": Transformer_Autoformer(input_dim, seq_len, horizon, patch_len=patch_len, d_model=d_model).to(device),
    "Informer": Transformer_Informer(input_dim, seq_len, horizon, patch_len=patch_len, d_model=d_model).to(device),
    "MLP": Koopman_MLP(input_dim=input_dim, seq_len=seq_len, horizon=horizon, d_model=d_model, hidden_dim=128, num_layers=3),
    "Autoencoder": MLP_Autoencoder(input_dim=input_dim, seq_len=seq_len, horizon=horizon, d_model=d_model, hidden_dim=128, num_layers=3)
}

results = {}

import time

true_vals = x_out.cpu().numpy()

for name, model in models.items():
    print(f"Training {name} ...")
    start = time.perf_counter()
    preds, loss, eigvals = train(model, x_in, x_out, epochs=epochs, koop_attr="koop")
    #results[name] = (preds, true_vals, eigvals)

    # Calculate errors / deviations
    error = preds - true_vals
    abs_error = np.abs(error)
    
    mae = np.mean(abs_error)
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    
    # Relative deviation (%)
    relative_error = abs_error / (np.abs(true_vals) + 1e-12)
    mean_relative_error = np.mean(relative_error) * 100
    
    # Maximum deviation
    max_error = np.max(abs_error)
    
    end = time.perf_counter()
    total_time = end - start
    print(f"Took {total_time:.6f} seconds")

    results[name] = {
    "preds": preds,
    "true_vals": true_vals,
    "eigvals": eigvals,
    "mae": mae,
    "rmse": rmse,
    "mean_relative_error": mean_relative_error,
    "max_error": max_error,
    "time": total_time
    }

# ============================================================
# Classical DMD baseline
# ============================================================

rank=None
ridge=1e-8

dmd_models = {
    "Classical DMD": ClassicalDMD(input_dim=input_dim, rank=rank, ridge=ridge),
    "Extended DMD": ExtendedDMD(input_dim=3, rank=rank, ridge=ridge, degree=2),
}

for name, model in dmd_models.items():
    print(f"Training {name} ...")
    start = time.perf_counter()
    preds, mse, eigvals, k = DMD_train(model, x_in, x_out)
    #results[name] = (preds, true_vals, eigvals)

    true_vals = x_out.cpu().numpy()

    # Calculate errors / deviations
    error = preds - true_vals
    abs_error = np.abs(error)
    
    mae = np.mean(abs_error)
    mse = mse[0]
    rmse = np.sqrt(mse)
    
    # Relative deviation (%)
    relative_error = abs_error / (np.abs(true_vals) + 1e-12)
    mean_relative_error = np.mean(relative_error) * 100
    
    # Maximum deviation
    max_error = np.max(abs_error)
    
    end = time.perf_counter()
    total_time = end - start
    print(f"Took {total_time:.6f} seconds")

    results[name] = {
    "preds": preds,
    "true_vals": true_vals,
    #"eigvals": eigvals,
    "mae": mae,
    "rmse": rmse,
    "mean_relative_error": mean_relative_error,
    "max_error": max_error,
    "time": total_time
    }

# === Save results ===
import os
import pickle

results_dir = "koopman_random_results"
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
    pickle.dump(results, f)
print(f"Results saved to {os.path.join(results_dir, 'results.pkl')}")


"""
# ==== PLOT PREDICTIONS ====
plt.figure(figsize=(16, 14))
titles = ["x", "y", "z"]

for idx in range(3):
    plt.subplot(3, 1, idx+1)
    plt.plot(results["PatchTST"][1][:, idx], label=f"True {titles[idx]}", color='black', linewidth=2)
    for name, (preds, _, _) in results.items():
        plt.plot(preds[:, idx], '--', label=f"{name} Pred {titles[idx]}", alpha=0.7)
    plt.legend()
    plt.title(f"Koopformer PRO Variants - {titles[idx]} Predictions")
    plt.grid()

plt.tight_layout()
plt.show()

# ==== PLOT Koopman Eigenvalue Evolution ====
plt.figure(figsize=(14, 6))

for name, (_, _, eigvals_hist) in results.items():
    plt.plot(np.arange(eigvals_hist.shape[0]), eigvals_hist.max(axis=1), label=name)

plt.xlabel("Training Step (x20 epochs)")
plt.ylabel("Max Eigenvalue (Spectral Radius)")
plt.title("Koopman Operator Stability Evolution")
plt.legend()
plt.grid()
plt.show()

"""



state_dim = 3

# --- Set consistent font sizes ---
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 12
SUPTITLE_FONTSIZE = 16
LEGEND_FONTSIZE = 12


# ==== PLOT PREDICTIONS ====

# ==== Plot Predictions ====
fig, axes = plt.subplots(state_dim, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
var_labels = [r"$x_1$", r"$x_2$", r"$x_3$"]



for idx, ax in enumerate(axes):
    ax.plot(results["PatchTST"]["true_vals"][:, idx], label=f"True {var_labels[idx]}", color='black', linewidth=2)
    for name, result in results.items():
        preds = result["preds"]
        ax.plot(preds[:, idx], '--', label=f"{name} Pred {var_labels[idx]}", alpha=0.7, linewidth=3)
    ax.set_ylabel(var_labels[idx], fontsize=LABEL_FONTSIZE)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, ncol=2)  # <--- legend in each subplot
    if idx == 0:
        pass
        #ax.set_title("Koopformer PRO Variants - Lorenz System State Predictions", fontsize=TITLE_FONTSIZE, pad=12)
    if idx == state_dim - 1:
        ax.set_xlabel("Time Step", fontsize=LABEL_FONTSIZE)

# Optionally, add a suptitle if you want a figure-wide title
# fig.suptitle("Koopformer PRO Variants - Rössler System State Predictions", fontsize=SUPTITLE_FONTSIZE, y=1.03)

plt.savefig("koopformer_rossler_variants_prediction.png", dpi=600, transparent=True)
plt.savefig("koopformer_rossler_variants_prediction.pdf", dpi=600)
plt.show()

print("\nExecution Time:")
print(f"{'Model':<22s}{'Time (s)':>16s}")

for name, r in results.items():
    train_time = r["time"]

    print(
        f"{name:<22s}"
        f"{train_time:>16.4f}"
    )

# ==== Plot Koopman Eigenvalue Evolution ====


import os
import matplotlib.pyplot as plt
import numpy as np

# Define your font sizes as before
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12

# Set line styles and colors for each model
model_styles = {
    "PatchTST":  {'linestyle': '-',  'color': '#1f77b4'},   # blue, solid
    "Autoformer":{'linestyle': '--', 'color': '#ff7f0e'},   # orange, dashed
    "Informer":  {'linestyle': '-.', 'color': '#2ca02c'},   # green, dash-dot
    "Classical DMD":  {'linestyle': '--', 'color': "#9619bd"},   # green, dash-dot
}

plt.figure(figsize=(8, 6))

for name, result in results.items():
    if "eigvals" in result:
        eigvals_hist = result["eigvals"]
        style = model_styles.get(name, {})

        eigvals_hist = np.asarray(eigvals_hist)

        if eigvals_hist.ndim == 1:
            # Only one set of eigenvalues: (16,)
            max_eig = np.max(np.abs(eigvals_hist))
            x = np.array([0])
            y = np.array([max_eig])

        elif eigvals_hist.ndim == 2:
            # Eigenvalue history: (training_steps, 16)
            x = np.arange(eigvals_hist.shape[0])
            y = np.max(np.abs(eigvals_hist), axis=1)

        else:
            raise ValueError(
                f"{name}: unexpected eigvals_hist shape "
                f"{eigvals_hist.shape}"
            )

        plt.plot(
            x,
            y,
            label=name,
            linestyle=style.get("linestyle", "-"),
            color=style.get("color", None),
            linewidth=2,
            marker="o" if len(x) == 1 else None,
        )

plt.xlabel("Training Step (x20 epochs)", fontsize=LABEL_FONTSIZE)
plt.ylabel("Max Eigenvalue (Spectral Radius)", fontsize=LABEL_FONTSIZE)
plt.title("Deep Koopman Operator Stability Evolution", fontsize=TITLE_FONTSIZE, pad=12)
plt.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True)
plt.tight_layout()
plt.savefig("koopformer_lorenz_variants_eigenvalues.png", dpi=600, transparent=True)
plt.savefig("koopformer_lorenz_variants_eigenvalues.pdf", dpi=600)
plt.show()


plt.figure(figsize=(12, 6))

for name, result in results.items():
    # Calculate prediction error
    error = np.asarray(result["preds"]) - np.asarray(result["true_vals"])

    # If predictions are multi-dimensional, average the
    # absolute error across all dimensions for each sample.
    if error.ndim > 1:
        error_to_plot = np.mean(
            np.abs(error),
            axis=tuple(range(1, error.ndim))
        )
    else:
        error_to_plot = np.abs(error)

    # Plot absolute error
    plt.plot(
        error_to_plot,
        label=name,
        linewidth=2
    )

plt.xlabel("Sample / Time Step")
plt.ylabel("Absolute Error")
plt.title("DMD Prediction Error")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Compare overall model errors
# ---------------------------------------------------------

names = list(results.keys())

mae_values = [
    results[name]["mae"]
    for name in names
]

rmse_values = [
    results[name]["rmse"]
    for name in names
]

# X positions for the bars
x = np.arange(len(names))
width = 0.35

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# MAE bars
ax.bar(
    x - width / 2,
    mae_values,
    width,
    label="MAE"
)

# RMSE bars
ax.bar(
    x + width / 2,
    rmse_values,
    width,
    label="RMSE"
)

# Labels and formatting
ax.set_xlabel("Model")
ax.set_ylabel("Error")
ax.set_title("DMD Model Error Comparison")

ax.set_xticks(x)
ax.set_xticklabels(names)

ax.legend()
ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()
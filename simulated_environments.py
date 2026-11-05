import argparse
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import sys
import os
import random


def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir

# Specify the GAMS system directory (Update this path according to your GAMS installation)
root_dir = setting_directory(1)  # Example path for Windows

print(root_dir)

from koopman.train_test import set_seed, train, DMD_train, evaluate_test, DMD_test
from koopman.backbone import SimpleLSTMForecaster, Transformer_PatchTST, Transformer_Informer, Transformer_Autoformer, ClassicalDMD, ExtendedDMD, MLP_Autoencoder, Koopman_MLP, Convolutional_AE
from koopman.dataset import build_dataset, build_dataset_scaled
from koopman.plot import plot_training, plot_per_feature
from koopman.plot import save_results

import pickle

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
def simulate_rossler_system(a=0.2, b=0.2, c=5.7, gamma=0.1, delta=0.2, dt=0.1, T=200.0):
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

# --------------------------------------------------------------------------- #
# 12)  Main loop                                                              #
# --------------------------------------------------------------------------- #
def main(file="outputs/old_flow.pkl", epochs=200, seq_len=200,
         patch_lens=None, horizons=None, indices=None,
         save_dir="results"):
    
    
    patch_lens = patch_lens or [80, 90, 100, 110, 120, 130]
    horizons   = horizons   or [10, 15, 20, 25, 30]
    indices    = indices    or [0, 500, 1000, 1500, 2000]
    d_model = 48

    save_dir   = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = save_dir / "metrics.csv"
    pd.DataFrame(columns=["Model", "PatchLen", "Horizon", "MSE", "MAE"]
        ).to_csv(metrics_file, index=False)
    
    data = simulate_random_ode_bounded(seed=111)
    print(data.shape)
    raw = data
    indices = [0]
    train_ratio = 0.8

    for patch_len in patch_lens:
        for horizon in horizons:
            print(f"\n>>> patch_len={patch_len}, horizon={horizon}\n")
            # ------------------------------------------------------------ #
            # keep seq_len = patch_len for variable-length experiments
            # If you need a *fixed* seq_len, replace `patch_len` below
            # with `seq_len`.
            # ------------------------------------------------------------ #
            x_in, x_out, test_in, test_out, scaler = build_dataset_scaled(raw[:600,:], indices,
                                           seq_len, horizon, train_ratio)

            # x_in = [N, H, F]
            # x_out = [N, H * F]
            F, H = len(indices), horizon
            res_plot = {}
            x_np = x_out.numpy()
            x_test_np = test_out.numpy()
            print(raw.shape, x_in.shape, x_out.shape, scaler)

            # ----- LSTM -----
            print("== LSTM ==")
            lstm = SimpleLSTMForecaster(F, 64, 2, horizon)
            pred, loss, eig = train(lstm, x_in, x_out,
                                    epochs=epochs, koop_attr=None, lyap_weight=0)
            pred_test, test_mse, test_mae = evaluate_test(lstm, test_in, test_out)

            res_plot["LSTM"] = {"pred": pred, "loss": loss, "eig": eig, "x_out": x_np,
                                "pred_test": pred_test, "y_test": x_test_np, "test_mse": test_mse, "test_mae": test_mae}
            prefix = save_dir / f"LSTM_{patch_len}_{horizon}"
            save_results(res_plot["LSTM"], prefix, metrics_file,
                         patch_len, horizon, d_model=d_model)
            torch.save(lstm.cpu().state_dict(), f"{prefix}.pt")

            # ----- PatchTST -----
            print("== PatchTST ==")
            m = Transformer_PatchTST(F, seq_len, horizon,
                                    patch_len=patch_len, d_model=d_model)
            pred, loss, eig = train(m, x_in, x_out, epochs=epochs, koop_attr="koop")
            pred_test, test_mse, test_mae = evaluate_test(m, test_in, test_out)

            res_plot["Transformer-PatchTST"] = {"pred": pred, "loss": loss, "eig": eig, "x_out": x_np,
                                "pred_test": pred_test, "y_test": x_test_np, "test_mse": test_mse, "test_mae": test_mae}
            prefix = save_dir / f"PatchTST_{patch_len}_{horizon}"
            save_results(res_plot["Transformer-PatchTST"], prefix, metrics_file,
                         patch_len, horizon, d_model=d_model)
            torch.save(m.cpu().state_dict(), f"{prefix}.pt")

            # ----- Autoformer -----
            print("== Autoformer ==")
            m = Transformer_Autoformer(F, seq_len, horizon,
                                      patch_len=patch_len, d_model=d_model)
            pred, loss, eig = train(m, x_in, x_out, epochs=epochs, koop_attr="koop")
            pred_test, test_mse, test_mae = evaluate_test(m, test_in, test_out)

            res_plot["Transformer-Autoformer"] = {"pred": pred, "loss": loss, "eig": eig, "x_out": x_np,
                                "pred_test": pred_test, "y_test": x_test_np, "test_mse": test_mse, "test_mae": test_mae}
            prefix = save_dir / f"Autoformer_{patch_len}_{horizon}"
            save_results(res_plot["Transformer-Autoformer"], prefix, metrics_file,
                         patch_len, horizon, d_model=d_model)
            torch.save(m.cpu().state_dict(), f"{prefix}.pt")

            # ----- Informer -----
            print("== Informer ==")
            m = Transformer_Informer(F, seq_len, horizon,
                                    patch_len=patch_len, d_model=d_model)
            pred, loss, eig = train(m, x_in, x_out, epochs=epochs, koop_attr="koop")
            pred_test, test_mse, test_mae = evaluate_test(m, test_in, test_out)

            res_plot["Transformer-Informer"] = {"pred": pred, "loss": loss, "eig": eig, "x_out": x_np,
                                "pred_test": pred_test, "y_test": x_test_np, "test_mse": test_mse, "test_mae": test_mae}
            prefix = save_dir / f"Informer_{patch_len}_{horizon}"
            save_results(res_plot["Transformer-Informer"], prefix, metrics_file,
                         patch_len, horizon, d_model=d_model)
            torch.save(m.cpu().state_dict(), f"{prefix}.pt")

            # ----- MLP -----
            print("== MLP ==")
            m = Koopman_MLP(input_dim=F, seq_len=seq_len, horizon=horizon, d_model=d_model, hidden_dim=128, num_layers=3)
            pred, loss, eig = train(m, x_in, x_out, epochs=epochs, koop_attr="koop")
            pred_test, test_mse, test_mae = evaluate_test(m, test_in, test_out)

            res_plot["MLP"] = {"pred": pred, "loss": loss, "eig": eig, "x_out": x_np,
                                "pred_test": pred_test, "y_test": x_test_np, "test_mse": test_mse, "test_mae": test_mae}

            prefix = save_dir / f"MLP{horizon}"
            save_results(
                res_plot["MLP"], prefix, metrics_file,
                        patch_len, horizon, d_model=d_model)
            torch.save(m.cpu().state_dict(), f"{prefix}.pt")

            # ----- Autoencoder -----
            print("== Autoencoder ==")
            m = MLP_Autoencoder(input_dim=F, seq_len=seq_len, horizon=horizon, d_model=d_model, hidden_dim=128, num_layers=3)
            pred, loss, eig = train(m, x_in, x_out, epochs=epochs, koop_attr="koop")
            pred_test, test_mse, test_mae = evaluate_test(m, test_in, test_out)

            res_plot["Autoencoder"] = {"pred": pred, "loss": loss, "eig": eig, "x_out": x_np,
                                "pred_test": pred_test, "y_test": x_test_np, "test_mse": test_mse, "test_mae": test_mae}

            prefix = save_dir / f"Autoencoder{horizon}"

            save_results(
                res_plot["Autoencoder"], prefix, metrics_file,
                        patch_len, horizon, d_model=d_model)
            torch.save(m.cpu().state_dict(), f"{prefix}.pt")

            rank=None
            ridge=1e-8

            # ----- Classical DMD -----
            print("== Classical DMD ==")

            dmd = ClassicalDMD(
                input_dim=F,
                rank=rank,
                ridge=ridge
            )

            # Train
            pred, loss, eigvals, K = DMD_train(dmd, x_in, x_out)

            # Test
            pred_test, test_mse, test_mae = DMD_test(dmd, test_in, test_out)
            
            # Store results
            res_plot["DMD"] = {"pred": pred, #"loss": loss, 
                               #"eig": eigvals, 
                               "x_out": x_np,
                                "pred_test": pred_test, "y_test": x_test_np, 
                                "test_mse": test_mse, "test_mae": test_mae}

            # Save results
            prefix = save_dir / f"DMD_{patch_len}_{horizon}"

            save_results(
                res_plot["DMD"],
                prefix,
                metrics_file,
                patch_len,
                horizon,
                d_model=None,
            )

            # Save DMD model
            torch.save(
                {
                    "K": K.cpu(),
                    "eigvals": eigvals,
                    "modes": (
                        dmd.modes.cpu()
                        if dmd.modes is not None
                        else None
                    ),
                    "rank": dmd.rank,
                    "ridge": dmd.ridge,
                },
                f"{prefix}.pt"
            )
            print(torch.mean((torch.from_numpy(pred) - x_out) ** 2).item())
            # ----- Extended DMD -----
            print("== Extended DMD ==")

            edmd = ExtendedDMD(
                input_dim=F,
                rank=rank,
                ridge=ridge,
                degree=2,
            )
            # Train
            pred, loss, eigvals, K = DMD_train(edmd, x_in, x_out)

            # Test
            pred_test, test_mse, test_mae = DMD_test(edmd, test_in, test_out)
            
            # Store results
            res_plot["ExtendedDMD"] = {"pred": pred, #"loss": loss, 
                                       #"eig": eigvals, 
                                       "x_out": x_np,
                                "pred_test": pred_test, "y_test": x_test_np, "test_mse": test_mse, "test_mae": test_mae}
            
            # Save results
            prefix = save_dir / f"ExtendedDMD_{patch_len}_{horizon}"

            save_results(
                res_plot["ExtendedDMD"],
                prefix,
                metrics_file,
                patch_len,
                horizon,
                d_model=None,
            )

            # Save Extended DMD model
            torch.save(
                {
                    "K": K.cpu(),
                    "eigvals": eigvals,
                    "modes": (
                        edmd.modes.cpu()
                        if edmd.modes is not None
                        else None
                    ),
                    "rank": edmd.rank,
                    "ridge": edmd.ridge,
                    "degree": edmd.degree,
                },
                f"{prefix}.pt"
            )

            # ----- Figures -----
            plot_training(res_plot,
                save_dir / f"train_patch{patch_len}_h{horizon}")
            
            # Training predictions
            plot_per_feature(
                x_np,
                {n: r["pred"] for n, r in res_plot.items()},
                F, H, -1,
                save_dir / f"train_feat_patch{patch_len}_h{horizon}", -0.1, 1.1
            )

            # Testing predictions
            plot_per_feature(
                x_test_np,
                {n: r["pred_test"] for n, r in res_plot.items()},
                F, H, -1,
                save_dir / f"test_feat_patch{patch_len}_h{horizon}", -0.1, 1.1
            )

            # Full summary
            print("\nFinal metrics:")
            print(
                f"{'Model':<22s}"
                f"{'Train MSE':>14s}"
                f"{'Train MAE':>14s}"
                f"{'Test MSE':>14s}"
                f"{'Test MAE':>14s}"
            )

            for name, r in res_plot.items():
                train_error = r["x_out"] - r["pred"]

                train_mse = np.mean(train_error ** 2)
                train_mae = np.mean(np.abs(train_error))

                test_mse = r.get("test_mse", float("nan"))
                test_mae = r.get("test_mae", float("nan"))

                print(
                    f"{name:<22s}"
                    f"{train_mse:>14.4e}"
                    f"{train_mae:>14.4e}"
                    f"{test_mse:>14.4e}"
                    f"{test_mae:>14.4e}"
                )


# --------------------------------------------------------------------------- #
# 13)  CLI                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Koopformer-PRO benchmark (HPC-ready)")
    p.add_argument("--file", type=str, default="")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=120)
    p.add_argument("--save_dir", type=str, default="results")
    args = p.parse_args()
    main(file=args.file, epochs=args.epochs,
         seq_len=args.seq_len, save_dir=args.save_dir, patch_lens=[50], horizons=[10])

import argparse
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import sys
import os



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



# --------------------------------------------------------------------------- #
#  Main loop                                                                  #
# --------------------------------------------------------------------------- #
def main(file="wind_speeds_2020.npy", epochs=200, seq_len=120,
         patch_lens=None, horizons=None, indices=None,
         save_dir="results"):
    
    
    patch_lens = patch_lens or [80, 90, 100, 110, 120, 130]
    horizons   = horizons   or [10, 15, 20, 25, 30]
    indices    = indices    or [0, 5000, 10000, 20000]
    d_model = 48

    save_dir   = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = save_dir / "metrics.csv"
    pd.DataFrame(columns=["Model", "PatchLen", "Horizon", "MSE", "MAE"]
        ).to_csv(metrics_file, index=False)

    raw = np.load(file)

    train_ratio = 0.8

    for patch_len in patch_lens:
        for horizon in horizons:
            print(f"\n>>> patch_len={patch_len}, horizon={horizon}\n")

            x_in, x_out, test_in, test_out, scaler = build_dataset_scaled(raw[:800,:], indices,
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Koopformer-PRO benchmark (HPC-ready)")
    p.add_argument("--file", type=str, default="")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=120)
    p.add_argument("--save_dir", type=str, default="results")
    args = p.parse_args()
    main(file=args.file, epochs=args.epochs,
         seq_len=args.seq_len, save_dir=args.save_dir, patch_lens=[50], horizons=[5])

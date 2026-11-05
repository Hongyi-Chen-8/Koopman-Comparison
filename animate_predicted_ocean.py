from copernicus.copernicus import *
import pickle
import numpy as np
from koopman.dataset import build_dataset

def check_features(raw):
    print(raw)

    print("Type:", type(raw))

    if isinstance(raw, np.ndarray):
        print("Shape:", raw.shape)
        print("Data type:", raw.dtype)
        print("First few values:")
        print(raw[:5])

    elif isinstance(raw, np.lib.npyio.NpzFile):
        print("Keys:", raw.files)

        for key in raw.files:
            data = raw[key]
            print(f"\n--- {key} ---")
            print("Shape:", data.shape)
            print("Data type:", data.dtype)
            print("First few values:")
            print(data[:5])
    return


pred = np.load(
    "results_full_boundary_not/MLP10_test_predictions.npy"
)

with open("outputs/old_flow.pkl", "rb") as f:
    raw = pickle.load(f)

raw_v = np.array([value['v'] for _, value in sorted(raw.items())])
raw_u = np.array([value['u'] for _, value in sorted(raw.items())])

raw_v = raw_v.reshape(raw_v.shape[0], -1)
raw_u = raw_u.reshape(raw_u.shape[0], -1)

comb_raw = np.stack((raw_v, raw_u), axis=-1)
print(comb_raw.shape)
comb_raw = comb_raw.reshape(comb_raw.shape[0], -1)
print(comb_raw.shape)
indices = list(range(comb_raw.shape[1]))

train_ratio = 0.8

x_in, x_out, test_in, pred, scaler = build_dataset(comb_raw[:600,:], indices,
                                        200, 10, train_ratio)

print(check_features(pred))

horizon = 10

# Original spatial grid
first_key = sorted(raw.keys())[0]
lon = np.asarray(raw[first_key]["lon"])
lat = np.asarray(raw[first_key]["lat"])

ny, nx = lon.shape

# Recover spatial grid and velocity components
pred = pred.reshape(-1, horizon, ny, nx, 2)

data_dict = {}

t = 0
data_dict = {}

for h in range(pred.shape[1]):
    data_dict[f"Horizon {h:02d}"] = {
        "lon": lon,
        "lat": lat,
        "u": pred[t, h, :, :, 1],
        "v": pred[t, h, :, :, 0],
    }

animate_ocean(
    data_dict,
    filename="ocean_prediction.mp4",
    fps=8,
)
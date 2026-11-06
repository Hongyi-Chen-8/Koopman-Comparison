import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def build_dataset_scaled(data: np.ndarray, indices: list[int], seq_len: int, horizon: int, train_ratio: float = 0.8,):
    sub = data[:, indices]

    split_idx = int(len(sub) * train_ratio)
    if split_idx <= seq_len:
        raise ValueError("Training set is too short for seq_len.")
    if len(sub) - split_idx < horizon:
        raise ValueError("Test set is too short for horizon.")

    scaler = MinMaxScaler().fit(sub[:split_idx])
    sub = scaler.transform(sub)

    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for i in range(split_idx - seq_len - horizon + 1):
        X_train.append(sub[i:i + seq_len])
        Y_train.append(
            sub[i + seq_len:i + seq_len + horizon].reshape(-1)
        )
    
    for target_start in range(split_idx, len(sub) - horizon + 1):
        i = target_start - seq_len
        X_test.append(sub[i:i + seq_len])
        Y_test.append(
            sub[target_start:target_start + horizon].reshape(-1)
        )

    x_train = torch.tensor(np.asarray(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.asarray(Y_train), dtype=torch.float32)

    x_test = torch.tensor(np.asarray(X_test), dtype=torch.float32)
    y_test = torch.tensor(np.asarray(Y_test), dtype=torch.float32)

    return x_train, y_train, x_test, y_test, scaler

def build_dataset(data: np.ndarray, indices: list[int], seq_len: int, horizon: int, train_ratio: float = 0.8,):
    sub = data[:, indices]

    split_idx = int(len(sub) * train_ratio)
    if split_idx <= seq_len:
        raise ValueError("Training set is too short for seq_len.")
    if len(sub) - split_idx < horizon:
        raise ValueError("Test set is too short for horizon.")

    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for i in range(split_idx - seq_len - horizon + 1):
        X_train.append(sub[i:i + seq_len])
        Y_train.append(
            sub[i + seq_len:i + seq_len + horizon].reshape(-1)
        )
    
    for target_start in range(split_idx, len(sub) - horizon + 1):
        i = target_start - seq_len
        X_test.append(sub[i:i + seq_len])
        Y_test.append(
            sub[target_start:target_start + horizon].reshape(-1)
        )

    x_train = torch.tensor(np.asarray(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.asarray(Y_train), dtype=torch.float32)

    x_test = torch.tensor(np.asarray(X_test), dtype=torch.float32)
    y_test = torch.tensor(np.asarray(Y_test), dtype=torch.float32)

    return x_train, y_train, x_test, y_test, 0

def inverse_scale(pred, scaler, horizon, n_features):
    pred = np.asarray(pred)
    single = pred.ndim == 1

    if single:
        pred = pred.reshape(1, -1)

    n_samples = pred.shape[0]

    pred = pred.reshape(n_samples, n_features, horizon)
    pred = pred.transpose(0, 2, 1)

    pred = scaler.inverse_transform(
        pred.reshape(-1, n_features)
    ).reshape(n_samples, horizon, n_features)

    return pred[0] if single else pred
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .layers import PatchEmbed1D, SinCosPosEnc, SeriesDecomp
from .koopman import StrictStableKoopmanOperator
from itertools import combinations_with_replacement

# --------------------------------------------------------------------------- #
# PatchTST backbone &                                                         #
# --------------------------------------------------------------------------- #
class PatchTST_Backbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int,
                 patch_len: int, d_model: int = 48,
                 num_layers: int = 3, num_heads: int = 4,
                 dim_ff: int = 96):
        super().__init__()
        self.patch = PatchEmbed1D(input_dim, d_model,
                                  patch_len, patch_len)
        n_patches = int(np.ceil(seq_len / patch_len))
        self.pos  = SinCosPosEnc(d_model, max_len=n_patches + 1)
        enc = nn.TransformerEncoderLayer(d_model, num_heads,
                                         dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        x   = self.patch(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.encoder(self.pos(x))
        return x[:, 0]


class Transformer_PatchTST(nn.Module):
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=48):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len,
                                          patch_len=patch_len,
                                          d_model=d_model)
        self.koop = StrictStableKoopmanOperator(d_model)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


# --------------------------------------------------------------------------- #
# Autoformer backbone                                                         #
# --------------------------------------------------------------------------- #
class SeriesDecomp(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.avg = nn.AvgPool1d(k, stride=1, padding=k // 2)

    def forward(self, x):
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        # --- make sure trend has the same length as x ---------------------- #
        if trend.size(1) != x.size(1):              # ### FIX 1 ###
            trend = trend[:, : x.size(1)]           # ### FIX 2 ###
        return trend, x - trend


class SimpleAutoformer(nn.Module):
    def __init__(self, input_len, horizon, input_dim,
                 patch_len,                # used as MA window
                 d_model=48, num_heads=4,
                 dim_ff=64, num_layers=3):
        super().__init__()
        self.dec   = SeriesDecomp(k=patch_len)
        self.embed = nn.Linear(input_dim, d_model)
        self.pos   = SinCosPosEnc(d_model, max_len=input_len)
        enc = nn.TransformerEncoderLayer(d_model, num_heads,
                                         dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.fc_seas  = nn.Linear(d_model, horizon)
        self.fc_trend = nn.Linear(input_len * input_dim, horizon)

    def forward(self, x):
        trend, seas = self.dec(x)
        seas   = self.encoder(self.pos(self.embed(seas)))
        seas_o = self.fc_seas(seas.mean(1))
        trend_o= self.fc_trend(trend.reshape(trend.size(0), -1))
        return seas_o + trend_o

class AutoformerBackbone(nn.Module):
    def __init__(self, input_len, d_model):
        super().__init__()
        self.decomp = SeriesDecomp(3)
        self.embed = nn.Linear(1, d_model)
        self.pos_encoding = SinCosPosEnc(d_model)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 2, 32, batch_first=True), 2)

    def forward(self, x):
        trend, seasonal = self.decomp(x)
        seasonal = self.pos_encoding(self.embed(seasonal))
        latent = self.encoder(seasonal).mean(dim=1)
        return latent

class Transformer_Autoformer(nn.Module):
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=16):
        super().__init__()
        self.backbone = SimpleAutoformer(seq_len, horizon, input_dim,
                                         patch_len=patch_len,
                                         d_model=d_model)
        self.koop = StrictStableKoopmanOperator(horizon)
        self.fc   = nn.Linear(horizon, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


# --------------------------------------------------------------------------- #
# Informer backbone                                                           #
# --------------------------------------------------------------------------- #
class InformerSparse(nn.Module):
    def __init__(self, input_dim, d_model=48, num_heads=4,
                 dim_ff=96, num_layers=3,
                 seq_len=120, patch_len=1):
        super().__init__()
        self.use_patch = patch_len > 1
        if self.use_patch:
            self.patch = PatchEmbed1D(input_dim, d_model,
                                      patch_len, patch_len)
            n_tokens = int(np.ceil(seq_len / patch_len))
        else:
            self.embed = nn.Linear(input_dim, d_model)
            n_tokens = seq_len
        self.pos  = SinCosPosEnc(d_model, max_len=n_tokens)
        enc = nn.TransformerEncoderLayer(d_model, num_heads,
                                         dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        if self.use_patch:
            x = self.patch(x)          # already d_model
        else:
            x = self.embed(x)
        x = self.encoder(self.pos(x))
        return x.mean(dim=1)


class Transformer_Informer(nn.Module):
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=48):
        super().__init__()
        self.backbone = InformerSparse(input_dim,
                                       d_model=d_model,
                                       seq_len=seq_len,
                                       patch_len=patch_len)
        self.koop = StrictStableKoopmanOperator(d_model)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


# --------------------------------------------------------------------------- #
# LSTM baseline                                                               #
# --------------------------------------------------------------------------- #
class SimpleLSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, horizon):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc   = nn.Linear(hidden_dim, horizon * input_dim)

    def forward(self, x, return_latents=False):
        _, (h_n, _) = self.lstm(x)
        z = h_n[-1]
        out = self.fc(z)
        if return_latents:
            return out, z, z
        return out

# --------------------------------------------------------------------------- #
# Classical SYD DMD                                                           #
# --------------------------------------------------------------------------- #

class ClassicalDMD:
    def __init__(self, input_dim, rank=None, ridge=1e-8):
        self.input_dim = input_dim
        self.rank = rank
        self.ridge = ridge

        self.K = None
        self.eigvals = None
        self.modes = None
        self.x_last = None

    @torch.no_grad()
    def fit(self, X):
        """
        Args:
            X: [batch_size, patch_len, input_dim]

        Returns:
            self
        """
        X = X.float()

        B, T, F = X.shape

        if F != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {F}"
            )

        if T < 2:
            raise ValueError("patch_len must be at least 2.")

        # Save the last state of each sequence
        self.x_last = X[:, -1, :]  # [B, F]

        # Consecutive snapshots
        X1 = X[:, :-1, :].transpose(1, 2)  # [B, F, T-1]
        X2 = X[:, 1:, :].transpose(1, 2)   # [B, F, T-1]

        # Batched SVD
        U, S, Vh = torch.linalg.svd(
            X1, full_matrices=False
        )

        # Optional truncated rank
        if self.rank is not None:
            r = min(self.rank, S.shape[-1])
            U = U[..., :r]
            S = S[..., :r]
            Vh = Vh[..., :r, :]

        V = Vh.transpose(-2, -1)

        # Regularized inverse singular values
        S_inv = S / (S.square() + self.ridge)

        # DMD operator: [B, F, F]
        self.K = (
            X2
            @ V
            @ torch.diag_embed(S_inv)
            @ U.transpose(-2, -1)
        )

        # Eigenvalues: [B, F]
        self.eigvals = torch.linalg.eigvals(self.K)

        return self

    @torch.no_grad()
    def predict(self, horizon):
        """
        Args:
            horizon: Number of future steps

        Returns:
            [batch_size, horizon * input_dim]
        """
        if self.K is None or self.x_last is None:
            raise RuntimeError("Call fit(X) before predict().")

        B, F = self.x_last.shape

        x = self.x_last
        predictions = []

        for _ in range(horizon):
            # x[t+1] = K @ x[t]
            x = torch.bmm(
                self.K, x.unsqueeze(-1)
            ).squeeze(-1)

            predictions.append(x)

        # [B, horizon, F] -> [B, horizon * F]
        return torch.stack(
            predictions, dim=1
        ).reshape(B, horizon * F)

# --------------------------------------------------------------------------- #
# SYD EDMD                                                                    #
# --------------------------------------------------------------------------- #
class ExtendedDMD:
    def __init__(self, input_dim, rank=None, ridge=1e-8, degree=2):
        self.input_dim = input_dim
        self.rank = rank
        self.ridge = ridge
        self.degree = degree

        self.K = None
        self.eigvals = None
        self.modes = None

        self.x_last = None
        self.Phi_last = None

    def lift(self, X):

        features = []

        # Constant
        features.append(
            torch.ones(
                X.shape[0], 1,
                device=X.device,
                dtype=X.dtype
            )
        )

        # Linear
        features.append(X)

        # Polynomial
        for degree in range(2, self.degree + 1):

            for indices in combinations_with_replacement(
                range(X.shape[1]),
                degree
            ):

                term = torch.ones(
                    X.shape[0],
                    device=X.device,
                    dtype=X.dtype
                )

                for i in indices:
                    term = term * X[:, i]

                features.append(term.unsqueeze(1))

        # Trigonometric dictionary
        features.append(torch.sin(X))
        features.append(torch.cos(X))

        # Exponential features
        #features.append(torch.exp(-X**2))

        # Absolute value
        features.append(torch.abs(X))

        return torch.cat(features, dim=1)


    @torch.no_grad()
    def fit(self, X):
        """
        Args:
            X: [batch_size, patch_len, input_dim]

        Returns:
            self
        """
        X = X.float()

        B, T, F = X.shape

        if F != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {F}"
            )

        if T < 2:
            raise ValueError("patch_len must be at least 2.")

        # Last observed state
        self.x_last = X[:, -1, :]

        Phi = self.lift(
            X.reshape(B * T, F)
        ).reshape(B, T, -1)

        Phi1 = Phi[:, :-1, :].transpose(1, 2)
        Phi2 = Phi[:, 1:, :].transpose(1, 2)

        # Batched SVD
        U, S, Vh = torch.linalg.svd(
            Phi1, full_matrices=False
        )

        # Optional truncated rank
        if self.rank is not None:
            r = min(self.rank, S.shape[-1])
            U = U[..., :r]
            S = S[..., :r]
            Vh = Vh[..., :r, :]

        V = Vh.transpose(-2, -1)

        # Regularized inverse singular values
        S_inv = S / (S.square() + self.ridge)

        # Koopman operator in lifted space
        self.K = (
            Phi2
            @ V
            @ torch.diag_embed(S_inv)
            @ U.transpose(-2, -1)
        )

        self.eigvals = torch.linalg.eigvals(self.K)

        return self

    @torch.no_grad()
    def predict(self, horizon):
        """
        Args:
            horizon: Number of future steps

        Returns:
            [batch_size, horizon * input_dim]
        """
        if self.K is None or self.x_last is None:
            raise RuntimeError("Call fit(X) before predict().")

        # Initial lifted state
        phi = self.lift(self.x_last)

        predictions = []

        for _ in range(horizon):
            # Advance one step in lifted space
            phi = torch.bmm(
                self.K, phi.unsqueeze(-1)
            ).squeeze(-1)

            # Recover the original state coordinates.
            # The first input_dim observables are X itself.
            x_pred = phi[:, 1:self.input_dim + 1]

            predictions.append(x_pred)

        # [B, horizon, F] -> [B, horizon * F]
        return torch.stack(
            predictions, dim=1
        ).reshape(
            self.x_last.shape[0], horizon * self.input_dim
        )

# --------------------------------------------------------------------------- #
# MLP Autoencoder                                                             #
# --------------------------------------------------------------------------- #

class MLP_Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        horizon: int,
        d_model: int = 48,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.d_model = d_model

        self.backbone = MLP_Backbone(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0,
        )

        # Latent linear Koopman evolution.
        # Initialize near identity for stable early training.
        self.koop = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.koop.weight)

        # Decoder reconstructs the future horizon.
        self.decoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * input_dim)
        )

    def forward(self, x, return_latents=False):
        # Encode history.
        z = self.backbone(x)

        # One learned latent transition.
        z_next = self.koop(z)

        # Decode into the forecast horizon.
        pred = self.decoder(z_next)
        pred = pred.reshape(
            x.size(0), self.horizon * self.input_dim
        )
        if return_latents:
            return pred, z, z_next

        return pred


class MLP_Backbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 48,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model

        layers = []
        in_features = seq_len * input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(
                    in_features if i == 0 else hidden_dim,
                    hidden_dim
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        layers.append(
            nn.Linear(
                hidden_dim if num_layers > 1 else in_features,
                d_model
            )
        )

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # Input: [batch, seq_len, input_dim]
        x = x.reshape(x.size(0), -1)

        # Output: [batch, d_model]
        z = self.encoder(x)

        return z
# --------------------------------------------------------------------------- #
# MLP Forcaster                                                             #
# --------------------------------------------------------------------------- #

class Koopman_MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        horizon,
        d_model=48,
        hidden_dim=128,
        num_layers=3,
        dropout=0.0,
    ):
        super().__init__()

        self.backbone = MLP_Backbone(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.koop = StrictStableKoopmanOperator(d_model)

        self.fc = nn.Linear(
            d_model,
            horizon * input_dim
        )

    def forward(self, x, return_latents=False):
        z = self.backbone(x)

        z_next, _, _ = self.koop(z)

        # Match your existing model output convention
        pred = self.fc(z_next).reshape(
            x.size(0),
            self.fc.out_features
        )

        if return_latents:
            return pred, z, z_next

        return pred

class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        height,
        width,
        d_model=48,
        hidden_dim=32,
    ):
        super().__init__()

        self.height = height
        self.width = width

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3,
                      stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3,
                      stride=2, padding=1),
            nn.GELU(),
        )

        # Infer the flattened feature dimension.
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, height, width)
            feature_dim = self.features(dummy).numel()

        self.projection = nn.Linear(feature_dim, d_model)

    def forward(self, x):
        # x: [B, C, H, W]
        features = self.features(x)
        return self.projection(features.flatten(start_dim=1))

# --------------------------------------------------------------------------- #
# CNN Koopman                                                                 #
# --------------------------------------------------------------------------- #
class ConvDecoder(nn.Module):
    def __init__(
        self,
        output_channels,
        height,
        width,
        d_model=48,
        hidden_dim=32,
    ):
        super().__init__()

        self.height = height
        self.width = width

        # Stride-2 encoder stages produce ceil(H/4), ceil(W/4).
        self.h0 = (height + 3) // 4
        self.w0 = (width + 3) // 4

        self.hidden_channels = hidden_dim * 2

        self.projection = nn.Linear(
            d_model,
            self.hidden_channels * self.h0 * self.w0,
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.hidden_channels,
                hidden_dim * 2,
                4, stride=2, padding=1,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                hidden_dim * 2,
                hidden_dim,
                4, stride=2, padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_channels, 3, padding=1),
        )

    def forward(self, z):
        # z: [B, d_model]
        x = self.projection(z)
        x = x.reshape(
            z.size(0),
            self.hidden_channels,
            self.h0,
            self.w0,
        )

        x = self.decoder(x)

        # Handle odd or non-multiple-of-four spatial dimensions.
        x = nn.functional.interpolate(
            x,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )

        return x

class TemporalEncoder(nn.Module):
    def __init__(
        self,
        d_model=48,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, z):
        # z: [B, S, d_model]
        _, h_n = self.gru(z)

        # Final temporal hidden state: [B, d_model]
        return h_n[-1]

class Convolutional_AE(nn.Module):
    def __init__(
        self,
        input_channels,
        height,
        width,
        horizon,
        d_model=48,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.horizon = horizon

        self.encoder = ConvEncoder(
            input_channels,
            height,
            width,
            d_model,
            hidden_dim,
        )

        self.temporal_encoder = TemporalEncoder(
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.koop = StrictStableKoopmanOperator(d_model)

        self.decoder = ConvDecoder(
            input_channels,
            height,
            width,
            d_model,
            hidden_dim,
        )

    def forward(self, x, return_latents=False):
        # x: [B, S, H, W, C]
        B, S, H, W, C = x.shape

        # Convert to [B, S, C, H, W]
        x = x.permute(0, 1, 4, 2, 3)

        # Encode all frames efficiently
        x = x.reshape(B * S, C, H, W)
        z_frames = self.encoder(x)

        # [B, S, d_model]
        z_frames = z_frames.reshape(B, S, -1)

        # Integrate temporal information from all S frames
        z = self.temporal_encoder(z_frames)

        # Forecast future latent states
        z_current = z
        predictions = []
        latents = []

        for _ in range(self.horizon):
            z_current, _, _ = self.koop(z_current)

            latents.append(z_current)
            predictions.append(self.decoder(z_current))

        # [B, horizon, C, H, W]
        pred = torch.stack(predictions, dim=1)

        # [B, horizon, H, W, C]
        pred = pred.permute(0, 1, 3, 4, 2)

        if return_latents:
            z_future = torch.stack(latents, dim=1)
            return pred, z, z_future

        return pred

    def reconstruct(self, x):
        """Reconstruct the last frame using the full input sequence."""
        # x: [B, S, H, W, C]
        B, S, H, W, C = x.shape

        x_chw = x.permute(0, 1, 4, 2, 3)
        x_chw = x_chw.reshape(B * S, C, H, W)

        z_frames = self.encoder(x_chw)
        z_frames = z_frames.reshape(B, S, -1)

        z = self.temporal_encoder(z_frames)
        recon = self.decoder(z)

        # [B, H, W, C]
        return recon.permute(0, 2, 3, 1)
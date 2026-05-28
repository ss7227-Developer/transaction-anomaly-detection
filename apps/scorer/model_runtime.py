from __future__ import annotations

import os
import joblib
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list[int], latent_dim: int):
        super().__init__()
        enc, d = [], input_dim
        for h in hidden_dim:
            enc += [nn.Linear(d, h), nn.ReLU(True)]
            d = h
        enc += [nn.Linear(d, latent_dim)]
        self.encoder = nn.Sequential(*enc)

        dec, d = [], latent_dim
        for h in reversed(hidden_dim):
            dec += [nn.Linear(d, h), nn.ReLU(True)]
            d = h
        dec += [nn.Linear(d, input_dim)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ModelRuntime:
    """
    Loads a versioned model bundle from model_dir:
      autoencoder_model.pt   — PyTorch weights
      minmax_scaler.joblib   — sklearn MinMaxScaler
      threshold.txt          — AE anomaly threshold
      isolation_forest.joblib — sklearn IsolationForest
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir

        # --- Autoencoder ---
        self.ae_threshold = float(
            open(os.path.join(model_dir, "threshold.txt")).read().strip()
        )
        self.scaler = joblib.load(os.path.join(model_dir, "minmax_scaler.joblib"))
        self._ae = Autoencoder(input_dim=7, hidden_dim=[64, 32, 16], latent_dim=8)
        self._ae.load_state_dict(
            torch.load(os.path.join(model_dir, "autoencoder_model.pt"), map_location="cpu")
        )
        self._ae.eval()

        # --- Isolation Forest ---
        self._if = joblib.load(os.path.join(model_dir, "isolation_forest.joblib"))

    @property
    def threshold(self) -> float:
        return self.ae_threshold

    def reconstruction_error(self, features_row: list[float]) -> float:
        x_s = self.scaler.transform([features_row])
        x = torch.tensor(x_s, dtype=torch.float32)
        with torch.no_grad():
            recon = self._ae(x)
            err = torch.mean((x - recon) ** 2, dim=1).item()
        return float(err)

    def if_predict(self, features_row: list[float]) -> tuple[float, bool]:
        """Returns (anomaly_score, is_anomaly). Score < 0 means more anomalous."""
        score = float(self._if.score_samples([features_row])[0])
        is_anomaly = self._if.predict([features_row])[0] == -1
        return score, is_anomaly

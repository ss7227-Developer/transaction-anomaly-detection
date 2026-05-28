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
    Loads a versioned model bundle:
      - autoencoder_model.pt
      - minmax_scaler.joblib
      - threshold.txt

    Produces reconstruction error on *scaled* features.
    """
    def __init__(self, model_dir: str):
        self.model_dir = model_dir

        threshold_path = os.path.join(model_dir, "threshold.txt")
        self.threshold = float(open(threshold_path).read().strip())

        scaler_path = os.path.join(model_dir, "minmax_scaler.joblib")
        self.scaler = joblib.load(scaler_path)

        weights_path = os.path.join(model_dir, "autoencoder_model.pt")
        self.model = Autoencoder(input_dim=7, hidden_dim=[64, 32, 16], latent_dim=8)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()

    def reconstruction_error(self, features_row: list[float]) -> float:
        """
        features_row: raw (unscaled) features in correct order.
        returns: mean squared reconstruction error in scaled space
        """
        x_scaled = self.scaler.transform([features_row])  # shape (1,7)
        x = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            recon = self.model(x)
            err = torch.mean((x - recon) ** 2, dim=1).item()

        return float(err)

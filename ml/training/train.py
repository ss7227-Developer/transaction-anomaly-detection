from __future__ import annotations

import os
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Must match inference architecture
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


def main():
    out_dir = "ml/artifacts/dev"
    os.makedirs(out_dir, exist_ok=True)

    # ---- Synthetic "normal" data (placeholder, realistic ranges) ----
    rng = np.random.default_rng(42)
    n = 30000
    X = np.zeros((n, 7), dtype=np.float32)

    X[:, 0] = rng.lognormal(mean=4.5, sigma=0.4, size=n)      # amount
    X[:, 1] = rng.integers(0, 24, size=n)                    # hour
    X[:, 2] = rng.integers(0, 7, size=n)                     # day
    X[:, 3] = np.abs(rng.normal(5.0, 3.0, size=n))           # distance
    X[:, 4] = np.abs(rng.normal(3600, 1800, size=n))         # time delta
    X[:, 5] = rng.normal(40.74, 0.05, size=n)                # lat
    X[:, 6] = rng.normal(-73.99, 0.05, size=n)               # lon

    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = Autoencoder(7, [64, 32, 16], 8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)

    model.train()
    for _ in range(12):
        opt.zero_grad()
        recon = model(X_train_t)
        loss = loss_fn(recon, X_train_t)
        loss.backward()
        opt.step()

    model.eval()
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    with torch.no_grad():
        recon = model(X_val_t)
        errs = torch.mean((X_val_t - recon) ** 2, dim=1).numpy()

    threshold = float(np.percentile(errs, 99.5))

    torch.save(model.state_dict(), f"{out_dir}/autoencoder_model.pt")
    joblib.dump(scaler, f"{out_dir}/minmax_scaler.joblib")
    with open(f"{out_dir}/threshold.txt", "w") as f:
        f.write(str(threshold))

    print("Artifacts written to", out_dir)
    print("Threshold:", threshold)


if __name__ == "__main__":
    main()

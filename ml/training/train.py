"""
Train both the Autoencoder and Isolation Forest on real simulated transaction data.

Usage:
    python -m ml.training.train --csv ml/artifacts/dev/simulated_transactions.csv --out-dir ml/artifacts/dev
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from ml.features.feature_engineering import FEATURE_NAMES

# --- Hyperparameters ---
INPUT_DIM = len(FEATURE_NAMES)   # 7
HIDDEN_DIM = [64, 32, 16]
LATENT_DIM = 8
NUM_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
THRESHOLD_PERCENTILE = 99.5      # AE threshold: 99.5th percentile of normal val errors
IF_CONTAMINATION = 0.005         # expected fraud rate (~0.5%)
IF_N_ESTIMATORS = 200


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


def load_and_prepare(csv_path: str):
    import pandas as pd

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    print(f"Loaded {len(df)} transactions | {df['is_fraud'].sum()} fraud ({df['is_fraud'].mean()*100:.2f}%)")

    # Feature engineering (mirrors feature_engineering.py for batch use)
    df["transaction_hour"] = df["timestamp"].dt.hour.astype(float)
    df["transaction_day_of_week"] = df["timestamp"].dt.dayofweek.astype(float)

    lat1 = np.radians(df["latitude"].values)
    lon1 = np.radians(df["longitude"].values)
    lat2 = np.radians(df["home_lat"].values)
    lon2 = np.radians(df["home_lon"].values)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["distance_from_home"] = 2 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    df["time_since_last_transaction"] = (
        df.groupby("user_id")["timestamp"].diff().dt.total_seconds().fillna(0)
    )
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def _train_autoencoder(X_train: np.ndarray) -> tuple[Autoencoder, MinMaxScaler]:
    scaler = MinMaxScaler()
    X_s = scaler.fit_transform(X_train)
    X_t = torch.tensor(X_s, dtype=torch.float32)

    model = Autoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_t), batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for (batch,) in loader:
            recon = model(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [AE] epoch {epoch+1:>3}/{NUM_EPOCHS}  loss={epoch_loss/len(loader):.6f}")

    return model, scaler


def _compute_ae_threshold(model: Autoencoder, scaler: MinMaxScaler, X_val: np.ndarray) -> float:
    model.eval()
    X_s = scaler.transform(X_val)
    X_t = torch.tensor(X_s, dtype=torch.float32)
    with torch.no_grad():
        recon = model(X_t)
        errs = torch.mean((X_t - recon) ** 2, dim=1).numpy()
    threshold = float(np.percentile(errs, THRESHOLD_PERCENTILE))
    print(f"  [AE] threshold (p{THRESHOLD_PERCENTILE}): {threshold:.8f}")
    return threshold


def _train_isolation_forest(X_train: np.ndarray) -> IsolationForest:
    model = IsolationForest(
        contamination=IF_CONTAMINATION,
        n_estimators=IF_N_ESTIMATORS,
        random_state=42,
    )
    model.fit(X_train)
    print(f"  [IF] trained on {len(X_train)} samples (contamination={IF_CONTAMINATION})")
    return model


def main(csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    df = load_and_prepare(csv_path)

    normal_X = df[df["is_fraud"] == False][FEATURE_NAMES].values
    fraud_X = df[df["is_fraud"] == True][FEATURE_NAMES].values

    X_train, X_val = train_test_split(normal_X, test_size=0.2, random_state=42)
    print(f"Split: {len(X_train)} normal train | {len(X_val)} normal val | {len(fraud_X)} fraud")

    # --- Autoencoder: trained on normal data only ---
    print("\n=== Training Autoencoder ===")
    ae_model, scaler = _train_autoencoder(X_train)
    threshold = _compute_ae_threshold(ae_model, scaler, X_val)

    torch.save(ae_model.state_dict(), os.path.join(out_dir, "autoencoder_model.pt"))
    joblib.dump(scaler, os.path.join(out_dir, "minmax_scaler.joblib"))
    with open(os.path.join(out_dir, "threshold.txt"), "w") as f:
        f.write(str(threshold))

    # --- Isolation Forest: trained on normal + known fraud to calibrate contamination ---
    print("\n=== Training Isolation Forest ===")
    X_if_train = np.vstack([X_train, fraud_X])
    if_model = _train_isolation_forest(X_if_train)
    joblib.dump(if_model, os.path.join(out_dir, "isolation_forest.joblib"))

    print(f"\nArtifacts saved to {out_dir}/")
    print("  autoencoder_model.pt")
    print("  minmax_scaler.joblib")
    print("  threshold.txt")
    print("  isolation_forest.joblib")
    print("\nNext: run ml/evaluation/evaluate.py to compare model performance.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="ml/artifacts/dev/simulated_transactions.csv")
    p.add_argument("--out-dir", default="ml/artifacts/dev")
    args = p.parse_args()
    main(args.csv, args.out_dir)

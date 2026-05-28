"""
Generate synthetic transaction data for model training and evaluation.

Usage:
    python apps/scripts/generate_data.py --users 100 --transactions 10000 --out ml/artifacts/dev/simulated_transactions.csv
"""
from __future__ import annotations

import argparse
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)


def _user_profiles(n: int) -> dict:
    profiles = {}
    for _ in range(n):
        uid = f"user_{str(uuid.uuid4())[:8]}"
        profiles[uid] = {
            "avg_amount": round(random.uniform(20, 150), 2),
            "std_amount": round(random.uniform(5, 30), 2),
            "home_lat": float(fake.latitude()),
            "home_lon": float(fake.longitude()),
            "payment_methods": random.sample(
                ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"],
                k=random.randint(1, 3),
            ),
        }
    return profiles


def _make_tx(uid: str, profile: dict, ts: datetime, is_fraud: bool, prev_ts: datetime | None) -> dict:
    amount = max(1.0, round(np.random.normal(profile["avg_amount"], profile["std_amount"]), 2))
    lat = profile["home_lat"] + random.uniform(-0.1, 0.1)
    lon = profile["home_lon"] + random.uniform(-0.1, 0.1)
    method = random.choice(profile["payment_methods"])
    merchant = fake.company()

    if is_fraud:
        fraud_type = random.choice(["high_amount", "foreign_location", "velocity", "stolen_card"])

        if fraud_type == "high_amount":
            amount = round(np.random.uniform(profile["avg_amount"] * 5, profile["avg_amount"] * 20), 2)

        elif fraud_type == "foreign_location":
            lat = float(fake.latitude())
            lon = float(fake.longitude())

        elif fraud_type == "velocity":
            amount = round(random.uniform(5, 50), 2)
            if prev_ts:
                ts = prev_ts + timedelta(seconds=random.randint(1, 10))

        elif fraud_type == "stolen_card":
            amount = round(np.random.uniform(profile["avg_amount"] * 2, profile["avg_amount"] * 10), 2)
            lat = float(fake.latitude())
            lon = float(fake.longitude())

    return {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": ts.isoformat(),
        "user_id": uid,
        "merchant": merchant,
        "amount": amount,
        "currency": "USD",
        "payment_method": method,
        "latitude": lat,
        "longitude": lon,
        "home_lat": profile["home_lat"],
        "home_lon": profile["home_lon"],
        "is_fraud": is_fraud,
    }


def generate(num_users: int, total_transactions: int, fraud_rate: float, start_date: str) -> pd.DataFrame:
    print(f"Generating {total_transactions} transactions for {num_users} users (fraud_rate={fraud_rate})...")
    profiles = _user_profiles(num_users)
    user_ids = list(profiles.keys())
    ts = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    last_ts: dict[str, datetime] = {uid: ts for uid in user_ids}

    rows = []
    i = 0
    while i < total_transactions:
        uid = random.choice(user_ids)
        profile = profiles[uid]
        is_fraud = random.random() < fraud_rate

        if is_fraud and random.random() < 0.3:
            # Velocity attack burst
            tx = _make_tx(uid, profile, ts, True, last_ts[uid])
            rows.append(tx)
            last_ts[uid] = datetime.fromisoformat(tx["timestamp"])
            i += 1
            for _ in range(random.randint(1, 3)):
                if i >= total_transactions:
                    break
                tx = _make_tx(uid, profile, ts, True, last_ts[uid])
                rows.append(tx)
                last_ts[uid] = datetime.fromisoformat(tx["timestamp"])
                i += 1
        else:
            tx = _make_tx(uid, profile, ts, is_fraud, last_ts[uid])
            rows.append(tx)
            last_ts[uid] = datetime.fromisoformat(tx["timestamp"])
            i += 1

        ts += timedelta(seconds=random.randint(1, 10))

        if i % (total_transactions // 10) == 0:
            print(f"  {i}/{total_transactions}...")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    fraud_count = df["is_fraud"].sum()
    print(f"Done. {len(df)} transactions | {fraud_count} fraud ({fraud_count/len(df)*100:.2f}%)")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=100)
    p.add_argument("--transactions", type=int, default=10000)
    p.add_argument("--fraud-rate", type=float, default=0.005)
    p.add_argument("--out", default="ml/artifacts/dev/simulated_transactions.csv")
    p.add_argument("--start-date", default="2023-01-01 00:00:00")
    args = p.parse_args()

    df = generate(args.users, args.transactions, args.fraud_rate, args.start_date)
    df.to_csv(args.out, index=False)
    print(f"Saved to {args.out}")

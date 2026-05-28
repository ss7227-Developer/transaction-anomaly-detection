from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

FEATURE_NAMES = [
    "amount",
    "transaction_hour",
    "transaction_day_of_week",
    "distance_from_home",
    "time_since_last_transaction",
    "latitude",
    "longitude",
]


@dataclass(frozen=True)
class Tx:
    transaction_id: str
    timestamp: str
    user_id: str
    amount: float
    latitude: float
    longitude: float
    home_lat: float
    home_lon: float


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2)
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_features(tx: Tx, last_ts: datetime | None) -> tuple[list[float], datetime]:
    ts = datetime.fromisoformat(tx.timestamp)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    distance = haversine_km(tx.latitude, tx.longitude, tx.home_lat, tx.home_lon)
    delta = 0.0 if last_ts is None else max(0.0, (ts - last_ts).total_seconds())

    feats = {
        "amount": float(tx.amount),
        "transaction_hour": float(ts.hour),
        "transaction_day_of_week": float(ts.weekday()),
        "distance_from_home": float(distance),
        "time_since_last_transaction": float(delta),
        "latitude": float(tx.latitude),
        "longitude": float(tx.longitude),
    }
    return [feats[n] for n in FEATURE_NAMES], ts

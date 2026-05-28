from __future__ import annotations

import os
import json
import time

import psycopg
import redis

from ml.features.feature_engineering import Tx, compute_features
from model_runtime import ModelRuntime
from online_state import OnlineState

DATABASE_URL = os.environ["DATABASE_URL"]
REDIS_URL = os.environ["REDIS_URL"]
MODEL_NAME = os.environ.get("MODEL_NAME", "autoencoder")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "dev")
MODEL_DIR = os.environ["MODEL_DIR"]

r = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def main():
    rt = ModelRuntime(MODEL_DIR)
    state = OnlineState(REDIS_URL)

    while True:
        try:
            msg = r.blpop("tx_queue", timeout=10)
            if not msg:
                continue

            _, payload = msg
            event = json.loads(payload)

            # ✅ IMPORTANT: must be a string, not a tuple
            client_id = str(event.get("client_id") or "demo").strip()

            tx = Tx(
                transaction_id=event["transaction_id"],
                timestamp=event["timestamp"],
                user_id=event["user_id"],
                amount=float(event["amount"]),
                latitude=float(event["latitude"]),
                longitude=float(event["longitude"]),
                home_lat=float(event["home_lat"]),
                home_lon=float(event["home_lon"]),
            )

            # ✅ tenant-scoped online state
            state_key = f"{client_id}:{tx.user_id}"
            last_ts = state.get_last_ts(state_key)
            features_row, current_ts = compute_features(tx, last_ts=last_ts)
            state.set_last_ts(state_key, current_ts)

            recon_error = rt.reconstruction_error(features_row)
            threshold = rt.threshold
            is_anom = recon_error > threshold

            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    # Upsert prediction (tenant-safe)
                    cur.execute(
                        """
                        insert into predictions(
                            client_id,
                            transaction_id, model_name, model_version,
                            reconstruction_error, anomaly_threshold, is_predicted_anomaly
                        )
                        values (%s,%s,%s,%s,%s,%s,%s)
                        on conflict (client_id, transaction_id, model_name, model_version)
                        do update set
                            reconstruction_error = excluded.reconstruction_error,
                            anomaly_threshold = excluded.anomaly_threshold,
                            is_predicted_anomaly = excluded.is_predicted_anomaly,
                            created_at = now()
                        """,
                        (
                            client_id,
                            tx.transaction_id,
                            MODEL_NAME,
                            MODEL_VERSION,
                            float(recon_error),
                            float(threshold),
                            bool(is_anom),
                        ),
                    )

                    # Create alert if anomaly (idempotent)
                    if is_anom:
                        cur.execute(
                            """
                            insert into alerts(client_id, transaction_id, status)
                            values (%s, %s, 'OPEN')
                            on conflict (client_id, transaction_id)
                            do nothing
                            """,
                            (client_id, tx.transaction_id),
                        )

                conn.commit()

            print(
                f"[scorer] tx={tx.transaction_id} client_id={client_id} "
                f"recon_error={recon_error:.8f} threshold={threshold:.8f} anomaly={is_anom}"
            )

        except Exception as e:
            print(f"[scorer] error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()

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

            state_key = f"{client_id}:{tx.user_id}"
            last_ts = state.get_last_ts(state_key)
            features_row, current_ts = compute_features(tx, last_ts=last_ts)
            state.set_last_ts(state_key, current_ts)

            # --- Autoencoder ---
            ae_error = rt.reconstruction_error(features_row)
            ae_threshold = rt.threshold
            ae_is_anom = ae_error > ae_threshold

            # --- Isolation Forest ---
            if_score, if_is_anom = rt.if_predict(features_row)

            # --- Ensemble: flag if either model flags ---
            is_anomaly = ae_is_anom or if_is_anom

            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    _upsert_prediction(cur, client_id, tx.transaction_id,
                                       "autoencoder", MODEL_VERSION,
                                       float(ae_error), float(ae_threshold), bool(ae_is_anom))

                    _upsert_prediction(cur, client_id, tx.transaction_id,
                                       "isolation_forest", MODEL_VERSION,
                                       float(if_score), 0.0, bool(if_is_anom))

                    if is_anomaly:
                        cur.execute(
                            """
                            insert into alerts(client_id, transaction_id, status)
                            values (%s, %s, 'OPEN')
                            on conflict (client_id, transaction_id) do nothing
                            """,
                            (client_id, tx.transaction_id),
                        )

                conn.commit()

            print(
                f"[scorer] tx={tx.transaction_id} client={client_id} "
                f"ae={ae_error:.6f}({'ANOM' if ae_is_anom else 'ok'}) "
                f"if={if_score:.4f}({'ANOM' if if_is_anom else 'ok'}) "
                f"=> {'ALERT' if is_anomaly else 'normal'}"
            )

        except Exception as e:
            print(f"[scorer] error: {e}")
            time.sleep(1)


def _upsert_prediction(cur, client_id, tx_id, model_name, model_version,
                        recon_error, threshold, is_anomaly):
    cur.execute(
        """
        insert into predictions(
            client_id, transaction_id, model_name, model_version,
            reconstruction_error, anomaly_threshold, is_predicted_anomaly
        )
        values (%s, %s, %s, %s, %s, %s, %s)
        on conflict (client_id, transaction_id, model_name, model_version)
        do update set
            reconstruction_error = excluded.reconstruction_error,
            anomaly_threshold    = excluded.anomaly_threshold,
            is_predicted_anomaly = excluded.is_predicted_anomaly,
            created_at           = now()
        """,
        (client_id, tx_id, model_name, model_version,
         recon_error, threshold, is_anomaly),
    )


if __name__ == "__main__":
    main()

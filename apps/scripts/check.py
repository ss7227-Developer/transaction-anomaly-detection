import os
import time
import json
import uuid
import requests
import psycopg


API_BASE = os.getenv("API_BASE", "http://localhost:8000")
CLIENT_ID = os.getenv("CLIENT_ID", "demo")
API_KEY = os.getenv("API_KEY", "demo_key_123")

# Optional: if you want DB verification too (recommended)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fraud:fraud@postgres:5432/fraud")

HEADERS = {
    "Content-Type": "application/json",
    "X-Client-Id": CLIENT_ID,
    "X-API-Key": API_KEY,
}


def wait_for_condition(fn, timeout_s=30, interval_s=1, label="condition"):
    start = time.time()
    while time.time() - start < timeout_s:
        val = fn()
        if val:
            return val
        time.sleep(interval_s)
    raise TimeoutError(f"Timed out waiting for {label} after {timeout_s}s")


def http_get(path):
    r = requests.get(f"{API_BASE}{path}", headers=HEADERS, timeout=10)
    return r.status_code, r.text


def http_post(path, payload):
    r = requests.post(f"{API_BASE}{path}", headers=HEADERS, data=json.dumps(payload), timeout=10)
    return r.status_code, r.text


def db_fetchone(query, params=None):
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            return cur.fetchone()


def db_fetchall(query, params=None):
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            return cur.fetchall()


def main():
    print("== Pre-AWS End-to-End Check ==")
    print(f"API_BASE={API_BASE}")
    print(f"CLIENT_ID={CLIENT_ID}")
    print()

    # 1) Health
    code, body = http_get("/health")
    assert code == 200, f"/health failed: {code} {body}"
    print("✅ API /health ok")

    # 2) Auth negative test (missing headers)
    r = requests.get(f"{API_BASE}/v1/alerts", timeout=10)  # no headers
    assert r.status_code in (401, 403), f"Expected 401/403 for missing auth, got {r.status_code} {r.text}"
    print("✅ Auth rejects missing headers")

    # 3) Ingest an extreme tx (should trigger anomaly -> alert)
    tx_id = str(uuid.uuid4())
    payload = {
        "transaction_id": tx_id,
        "user_id": "preaws_acct_1",
        "amount": 99999,
        "latitude": 34.0522,
        "longitude": -118.2437,
        "home_lat": 40.7128,
        "home_lon": -74.0060,
    }
    code, body = http_post("/v1/transactions", payload)
    assert code in (200, 202), f"POST /v1/transactions failed: {code} {body}"
    print(f"✅ Transaction enqueued tx_id={tx_id}")

    # 4) DB: transaction exists (optional but very useful)
    tx_row = wait_for_condition(
        lambda: db_fetchone(
            "select id, client_id, user_id, amount from transactions where id=%s",
            (tx_id,),
        ),
        timeout_s=20,
        interval_s=1,
        label="transaction row",
    )
    assert tx_row[1] == CLIENT_ID, f"Transaction client_id mismatch. Expected {CLIENT_ID}, got {tx_row[1]}"
    print("✅ Transaction persisted in Postgres")

    # 5) DB: prediction exists
    pred_row = wait_for_condition(
        lambda: db_fetchone(
            """
            select client_id, transaction_id, is_predicted_anomaly, reconstruction_error, anomaly_threshold
            from predictions
            where transaction_id=%s and client_id=%s
            order by created_at desc
            limit 1
            """,
            (tx_id, CLIENT_ID),
        ),
        timeout_s=60,
        interval_s=2,
        label="prediction row",
    )
    print(f"✅ Prediction written is_anomaly={pred_row[2]} recon_err={pred_row[3]} threshold={pred_row[4]}")
    assert pred_row[2] is True, "Expected anomaly=True for extreme tx"

    # 6) DB: alert exists
    alert_row = wait_for_condition(
        lambda: db_fetchone(
            "select client_id, transaction_id, status from alerts where transaction_id=%s and client_id=%s",
            (tx_id, CLIENT_ID),
        ),
        timeout_s=30,
        interval_s=2,
        label="alert row",
    )
    print(f"✅ Alert created status={alert_row[2]}")
    assert alert_row[2] == "OPEN", f"Expected OPEN, got {alert_row[2]}"

    # 7) API: list alerts returns it
    def alert_in_api():
        code, body = http_get("/v1/alerts?status=OPEN&limit=20")
        if code != 200:
            return None
        try:
            data = json.loads(body)
        except Exception:
            return None
        for a in data:
            if a.get("transaction_id") == tx_id:
                return a
        return None

    alert_obj = wait_for_condition(alert_in_api, timeout_s=30, interval_s=2, label="alert visible via API")
    print("✅ Alerts API returns the new alert")

    # 8) API: update status
    code, body = http_post(f"/v1/alerts/{tx_id}/status", {"status": "INVESTIGATING"})
    assert code == 200, f"Update status failed: {code} {body}"
    print("✅ Alert status updated to INVESTIGATING")

    # 9) API: add note
    code, body = http_post(f"/v1/alerts/{tx_id}/note", {"note": "Pre-AWS smoke test: anomaly confirmed."})
    assert code == 200, f"Update note failed: {code} {body}"
    print("✅ Alert note updated")

    # 10) DB: final status reflects updates
    final_row = db_fetchone(
        "select status, note from alerts where transaction_id=%s and client_id=%s",
        (tx_id, CLIENT_ID),
    )
    assert final_row[0] == "INVESTIGATING", f"Expected INVESTIGATING, got {final_row[0]}"
    assert "smoke test" in (final_row[1] or "").lower(), "Note not persisted"
    print("✅ Postgres reflects status + note")

    print("\n🎉 All checks passed. You’re ready for AWS (EC2 + containers).")


if __name__ == "__main__":
    main()

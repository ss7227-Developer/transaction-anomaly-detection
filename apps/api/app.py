from __future__ import annotations

import os
import json
import uuid
from datetime import datetime, timezone

import psycopg
import redis
from flask import Flask, request, jsonify

DATABASE_URL = os.environ["DATABASE_URL"]
REDIS_URL = os.environ["REDIS_URL"]

r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
app = Flask(__name__)


def _parse_client_keys(s: str) -> dict[str, str]:
    out = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


CLIENT_API_KEYS = _parse_client_keys(os.environ.get("CLIENT_API_KEYS", ""))


def _require_auth(req) -> str:
    client_id = req.headers.get("X-Client-Id")
    api_key = req.headers.get("X-API-Key")
    if not client_id or not api_key:
        raise ValueError("missing_auth_headers")
    expected = CLIENT_API_KEYS.get(client_id)
    if not expected or expected != api_key:
        raise ValueError("invalid_api_key")
    return client_id


def _row(row, cols):
    return {c: row[i] for i, c in enumerate(cols)}


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/v1/transactions")
def ingest_transaction():
    body = request.get_json(force=True)

    try:
        client_id = _require_auth(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 401

    required = ["user_id", "amount", "latitude", "longitude", "home_lat", "home_lon"]
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": "missing_fields", "fields": missing}), 400

    tx_id = body.get("transaction_id") or str(uuid.uuid4())
    ts = body.get("timestamp") or datetime.now(timezone.utc).isoformat()

    event = {
        "client_id": client_id,
        "transaction_id": tx_id,
        "timestamp": ts,
        "user_id": body["user_id"],
        "amount": float(body["amount"]),
        "currency": body.get("currency", "USD"),
        "merchant": body.get("merchant"),
        "payment_method": body.get("payment_method"),
        "latitude": float(body["latitude"]),
        "longitude": float(body["longitude"]),
        "home_lat": float(body["home_lat"]),
        "home_lon": float(body["home_lon"]),
    }

    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(
            """
            insert into transactions(id, client_id, user_id, ts, amount, currency,
                                     merchant, payment_method, latitude, longitude,
                                     home_lat, home_lon)
            values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            on conflict (id) do nothing
            """,
            (tx_id, client_id, event["user_id"], ts, event["amount"], event["currency"],
             event["merchant"], event["payment_method"],
             event["latitude"], event["longitude"], event["home_lat"], event["home_lon"]),
        )
        conn.commit()

    r.rpush("tx_queue", json.dumps(event))
    return jsonify({"transaction_id": tx_id, "enqueued": True}), 202


@app.get("/v1/transactions/<tx_id>")
def get_transaction(tx_id: str):
    try:
        client_id = _require_auth(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 401

    cols = ["id", "client_id", "user_id", "ts", "amount", "currency",
            "merchant", "payment_method", "latitude", "longitude",
            "home_lat", "home_lon", "created_at"]

    with psycopg.connect(DATABASE_URL) as conn:
        cur = conn.execute(
            "select " + ", ".join(cols) +
            " from transactions where id = %s and client_id = %s",
            (tx_id, client_id),
        )
        row = cur.fetchone()

    if not row:
        return jsonify({"error": "not_found"}), 404
    return jsonify(_row(row, cols))


@app.get("/v1/predictions")
def list_predictions():
    try:
        client_id = _require_auth(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 401

    model_name = request.args.get("model_name")
    limit = request.args.get("limit", "50")
    try:
        limit_i = max(1, min(200, int(limit)))
    except Exception:
        return jsonify({"error": "invalid_limit"}), 400

    cols = ["client_id", "transaction_id", "model_name", "model_version",
            "reconstruction_error", "anomaly_threshold", "is_predicted_anomaly", "created_at"]

    sql = ("select " + ", ".join(cols) +
           " from predictions where client_id = %s")
    params: list = [client_id]

    if model_name:
        sql += " and model_name = %s"
        params.append(model_name)

    sql += " order by created_at desc limit %s"
    params.append(limit_i)

    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()

    return jsonify([_row(r, cols) for r in rows])


@app.get("/v1/alerts")
def list_alerts():
    try:
        client_id = _require_auth(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 401

    status = request.args.get("status")
    limit = request.args.get("limit", "50")
    try:
        limit_i = max(1, min(200, int(limit)))
    except Exception:
        return jsonify({"error": "invalid_limit"}), 400

    sql = ("select client_id, transaction_id, status, note, created_at, updated_at"
           " from alerts where client_id = %s")
    params: list = [client_id]

    if status:
        sql += " and status = %s"
        params.append(status)

    sql += " order by updated_at desc limit %s"
    params.append(limit_i)

    cols = ["client_id", "transaction_id", "status", "note", "created_at", "updated_at"]
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()

    return jsonify([_row(r, cols) for r in rows])


@app.post("/v1/alerts/<tx_id>/status")
def update_alert_status(tx_id: str):
    try:
        client_id = _require_auth(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 401

    body = request.get_json(force=True)
    status = body.get("status")
    if status not in {"OPEN", "INVESTIGATING", "RESOLVED"}:
        return jsonify({"error": "invalid_status"}), 400

    cols = ["client_id", "transaction_id", "status", "note", "created_at", "updated_at"]
    with psycopg.connect(DATABASE_URL) as conn:
        cur = conn.execute(
            """
            update alerts set status = %s
            where client_id = %s and transaction_id = %s
            returning client_id, transaction_id, status, note, created_at, updated_at
            """,
            (status, client_id, tx_id),
        )
        row = cur.fetchone()
        conn.commit()

    if not row:
        return jsonify({"error": "not_found"}), 404
    return jsonify(_row(row, cols))


@app.post("/v1/alerts/<tx_id>/note")
def update_alert_note(tx_id: str):
    try:
        client_id = _require_auth(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 401

    body = request.get_json(force=True)
    note = body.get("note")
    if note is None or not isinstance(note, str) or len(note) > 2000:
        return jsonify({"error": "invalid_note"}), 400

    cols = ["client_id", "transaction_id", "status", "note", "created_at", "updated_at"]
    with psycopg.connect(DATABASE_URL) as conn:
        cur = conn.execute(
            """
            update alerts set note = %s
            where client_id = %s and transaction_id = %s
            returning client_id, transaction_id, status, note, created_at, updated_at
            """,
            (note, client_id, tx_id),
        )
        row = cur.fetchone()
        conn.commit()

    if not row:
        return jsonify({"error": "not_found"}), 404
    return jsonify(_row(row, cols))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

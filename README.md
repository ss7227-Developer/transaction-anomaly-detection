# Real-Time Transaction Anomaly Detection

Event-driven fraud detection system that scores financial transactions using an **ensemble of two unsupervised ML models** — a PyTorch autoencoder and an Isolation Forest — served via a Flask API with Redis queuing, PostgreSQL audit storage, multi-tenant API key authentication, and an alert management workflow.

## Architecture

```
apps/
  api/          Flask REST API — authenticates clients, persists raw txns, enqueues to Redis
  scorer/       Async worker — dequeues, runs ensemble inference, writes predictions + alerts
  scripts/      Data generation and load testing utilities
ml/
  features/     Feature engineering (haversine distance, time delta, temporal features)
  training/     Training pipeline for both models using real simulated data
  evaluation/   Comparative evaluation: F1, precision, recall, confusion matrix, business impact
  artifacts/    Saved model weights, scaler, threshold (gitignored)
infra/
  postgres/     DB schema — transactions, per-model predictions, alert lifecycle
  docker-compose.yml
```

## ML Design

### Models

Two unsupervised anomaly detectors are trained and run in ensemble — a transaction is flagged if **either** model raises an alert.

| Model | Trained on | Anomaly signal |
|---|---|---|
| **Autoencoder** (PyTorch) | Normal transactions only | Reconstruction error > learned threshold |
| **Isolation Forest** (scikit-learn) | Normal + known fraud | `predict() == -1` (isolated outlier) |

The autoencoder learns a compressed latent representation of normal spending patterns. Unusual transactions reconstruct poorly, producing high MSE error. The Isolation Forest provides a complementary tree-based signal that doesn't require calibrating a continuous threshold.

### Features (7 total)

| Feature | Source |
|---|---|
| `amount` | Raw transaction field |
| `transaction_hour` | Extracted from timestamp |
| `transaction_day_of_week` | Extracted from timestamp |
| `distance_from_home` | Haversine distance (km) between tx location and user home |
| `time_since_last_transaction` | Per-user velocity; tracked in Redis per tenant across requests |
| `latitude`, `longitude` | Raw GPS fields |

### Fraud types modeled in training data

- **High-amount**: amount 5–20× user average
- **Foreign location**: random GPS far from home
- **Velocity attack**: 1–4 rapid small transactions in burst
- **Stolen card**: high amount + foreign location

### Evaluation results

Trained and evaluated on 10,000 simulated transactions (90 fraud, 0.9% rate) across 100 users.
Test set: 1,982 normal + 90 fraud held-out samples.

| Model | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| Autoencoder | **0.75** | 0.86 | 0.67 | 1972 | 10 | 30 | 60 |
| Isolation Forest | 0.54 | **0.92** | 0.38 | 1979 | 3 | 56 | 34 |
| Ensemble (AE OR IF) | 0.74 | 0.83 | 0.67 | 1970 | 12 | 30 | 60 |

**Simulated business impact** ($1,000 saved per fraud caught, $100 cost per false alarm):

| Model | Net Savings |
|---|---|
| Autoencoder | **$59,000** |
| Isolation Forest | $33,700 |
| Ensemble (AE OR IF) | $58,800 |

The Autoencoder is the strongest single model. The Isolation Forest is highly precise (92%) but misses 62% of fraud — useful as a high-confidence signal. The ensemble marginally increases false positives without improving recall, suggesting AE alone is the better production choice at this threshold.

Business impact uses simulated costs: $100/false positive (manual review), $1,000/true positive (fraud prevented), $500 average undetected fraud loss.

## Stack

- **ML**: PyTorch autoencoder + scikit-learn Isolation Forest
- **API**: Flask + psycopg3, multi-tenant API key auth
- **Queue**: Redis (async decoupling between ingestion and scoring)
- **Storage**: PostgreSQL — immutable transaction log, per-model predictions, alert lifecycle
- **Infra**: Docker Compose (API, scorer worker, Redis, PostgreSQL)

## Running Locally

```bash
# 1. Generate training data
python apps/scripts/generate_data.py \
  --users 100 --transactions 10000 \
  --out ml/artifacts/dev/simulated_transactions.csv

# 2. Train both models
python -m ml.training.train \
  --csv ml/artifacts/dev/simulated_transactions.csv \
  --out-dir ml/artifacts/dev

# 3. Evaluate and compare models
python -m ml.evaluation.evaluate \
  --csv ml/artifacts/dev/simulated_transactions.csv \
  --model-dir ml/artifacts/dev

# 4. Start all services
docker-compose -f infra/docker-compose.yml up --build
```

## API

All endpoints require `X-Client-Id` and `X-API-Key` headers. Set clients via the `CLIENT_API_KEYS` env var:

```
CLIENT_API_KEYS=client_a=secret1,client_b=secret2
```

### Ingest a transaction
```bash
curl -X POST http://localhost:8000/v1/transactions \
  -H "X-Client-Id: client_a" \
  -H "X-API-Key: secret1" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_17",
    "amount": 9500,
    "latitude": 51.5,
    "longitude": -0.1,
    "home_lat": 40.71,
    "home_lon": -74.01
  }'
```

### Query predictions (filter by model)
```bash
curl "http://localhost:8000/v1/predictions?model_name=autoencoder&limit=20" \
  -H "X-Client-Id: client_a" -H "X-API-Key: secret1"

curl "http://localhost:8000/v1/predictions?model_name=isolation_forest&limit=20" \
  -H "X-Client-Id: client_a" -H "X-API-Key: secret1"
```

### Manage alerts
```bash
# List open alerts
curl "http://localhost:8000/v1/alerts?status=OPEN" \
  -H "X-Client-Id: client_a" -H "X-API-Key: secret1"

# Update status
curl -X POST "http://localhost:8000/v1/alerts/<tx_id>/status" \
  -H "X-Client-Id: client_a" -H "X-API-Key: secret1" \
  -d '{"status": "INVESTIGATING"}'

# Add analyst note
curl -X POST "http://localhost:8000/v1/alerts/<tx_id>/note" \
  -H "X-Client-Id: client_a" -H "X-API-Key: secret1" \
  -d '{"note": "Confirmed fraud — user contacted."}'
```

## Model Governance

- Each prediction row records `model_name` and `model_version` alongside the score — the ensemble decision is auditable per model
- AE threshold is computed at training time (99.5th percentile of normal validation errors) and versioned with the model weights
- All transactions are written to an immutable PostgreSQL log before scoring begins
- Per-user Redis state (last transaction timestamp) is tenant-scoped (`client_id:user_id`) with a 30-day TTL

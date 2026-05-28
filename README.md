# Real-Time Transaction Anomaly Detection

Event-driven risk monitoring system that scores financial transactions with low latency using a PyTorch autoencoder, Flask API, and Redis queue — with full audit logging and model versioning.

## Architecture

```
apps/
  api/        Flask REST API — receives transactions, enqueues to Redis
  scorer/     Async worker — dequeues, runs autoencoder inference, writes to PostgreSQL
  scripts/    Utility scripts (data checks, load testing)
ml/
  training/   Autoencoder training pipeline (PyTorch)
  features/   Feature engineering and preprocessing
  artifacts/  Saved model weights and scaler (gitignored — see below)
infra/
  postgres/   DB schema (init.sql)
  docker-compose.yml
```

## Stack

- **ML**: PyTorch autoencoder trained on normalized transaction features
- **API**: Flask + Redis queue for async scoring
- **Storage**: PostgreSQL with append-only audit log table
- **Infra**: Docker Compose (API, scorer worker, Redis, PostgreSQL)

## Running Locally

```bash
# Start all services
docker-compose -f infra/docker-compose.yml up --build

# Train the model (saves artifacts to ml/artifacts/dev/)
cd ml && python training/train.py

# Send a test transaction
curl -X POST http://localhost:5000/score \
  -H "Content-Type: application/json" \
  -d '{"amount": 9500, "merchant_id": "m_42", "user_id": "u_17"}'
```

## Model

The autoencoder learns a compressed representation of normal transaction patterns. At inference time, reconstruction error above a learned threshold flags a transaction as anomalous. Threshold is persisted alongside the model weights for reproducibility.

## Governance

- Model version is logged with every scored transaction
- All scores are written to an immutable audit log in PostgreSQL
- Scaler and threshold are versioned alongside model weights

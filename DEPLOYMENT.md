# Production Deployment Guide

**Last Updated:** January 2, 2026
**Architecture:** 365-Day Predictions with Model Versioning and PyTorch Support

## Quick Start

1. Install: `pip install -r requirements.txt`
2. Migrate: `python scripts/migrate_db.py`
3. Train & Register: `python scripts/batch_register.py --date-tag 2026-01-02`
4. Predict: `python scripts/cron_predict_new.py`
5. API: `uvicorn backend.app:app --reload`

---

## Table of Contents

1. [Setup & Migrations](#setup--migrations)
2. [Data & Workflows](#data--workflows)
3. [Model Training & Versioning](#model-training--versioning)
4. [API Endpoints](#api-endpoints)
5. [Local Development](#local-development)
6. [Render Deployment](#render-deployment)
7. [Troubleshooting](#troubleshooting)

---

## Setup & Migrations

### Database Configuration

Configure ackend/database.ini:
`ini
[postgresql]
host = localhost
port = 5432
database = stock_predictor
user = postgres
password = your_password
`

### Run Migrations

`ash
python scripts/migrate_db.py
`

This script creates or updates:
- `model_versions` table for tracking trained models
- `predictions` table with `model_version_id` foreign key
- `trading_day_offset` column for 365-day forecasting
- Appropriate indexes and constraints

### Download Initial Data

`ash
python scripts/download_data.py
`

Fetches OHLCV data from yfinance for all tickers.

---

## Data & Workflows

### Daily Data Update (9:30 AM ET)

`ash
python scripts/update_data.py
`

- Downloads new market data since last update
- Saves to `stock_prices` table in PostgreSQL

### Daily Prediction Generation (4:00 PM+ ET)

`ash
python scripts/cron_predict_new.py
`

- Loads latest `ModelVersion` (by `trained_at` timestamp)
- For each ticker: generates 365 trading-day forecasts
- Stores predictions in `predictions` table

**Supports both model types:**
- **sklearn** (RandomForest): Uses last row for inference
- **PyTorch** (HCLA): Uses last 30-row sequence window for inference

---

## Model Training & Versioning

### Train Models

`ash
python stock_models/ford_model.py
python stock_models/tsla_model.py
# ... train other tickers
`

Each script saves artifacts to `saved_models/<ticker>_artifacts.joblib`.

### Register Versions

**Option A: Register one model**

`ash
python scripts/register_model.py \
  --version v2026-01-02 \
  --model-type RandomForest \
  --artifact-path saved_models/ford_artifacts.joblib
`

**Option B: Batch register all artifacts**

`ash
python scripts/batch_register.py --date-tag 2026-01-02
`

Auto-discovers artifacts and creates ticker-specific version tags:
- `f_v1_2026-01-02` (Ford)
- `tsla_v1_2026-01-02` (Tesla)
- etc.

### Automated Retraining

`ash
python scripts/retrain_all.py --date-tag 2026-01-02
`

- Runs all training scripts
- Auto-registers artifacts
- Generates predictions

---

## API Endpoints

### GET /predictions/{ticker}

Returns 365 predictions for latest model version.

**Optional query parameter:** `model_version_id=1`

**Response:**
`json
{
  "ticker": "TSLA",
  "model_version_id": 1,
  "model_version_tag": "tsla_v1_2026-01-02",
  "predictions": [
    {"target_date": "2026-01-05", "predicted_price": 245.32, "trading_day_offset": 1},
    {"target_date": "2026-01-06", "predicted_price": 246.15, "trading_day_offset": 2},
    ...
  ]
}
`

### GET /

Health check: `{"status": "ok"}`

### GET /docs

Swagger API documentation.

---

## Local Development

### Setup

`ash
git clone <repo>
cd stock_predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/migrate_db.py
`

### Train & Register

`ash
python stock_models/ford_model.py
python scripts/batch_register.py --date-tag 2026-01-02
python scripts/cron_predict_new.py
`

### Start API

`ash
cd backend
uvicorn app:app --reload
`

Browse: http://127.0.0.1:8000/docs

---

## Render Deployment

### Steps

1. Connect GitHub repo to Render
2. Set **Build Command**: `pip install -r requirements.txt`
3. Set **Start Command**: `uvicorn backend.app:app --host 0.0.0.0 --port \`
4. Add `DATABASE_URL` env variable (PostgreSQL connection string)
5. Configure cron jobs (see render.yaml template below)

### Cron Jobs (render.yaml)

`yaml
services:
  - type: web
    name: stock-predictor-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn backend.app:app --host 0.0.0.0 --port \

cronJobs:
  - name: daily-data-update
    schedule: "30 14 * * 1-5"  # 2:30 PM UTC = 9:30 AM ET
    command: python scripts/update_data.py
  
  - name: daily-predictions
    schedule: "0 21 * * 1-5"   # 9:00 PM UTC = 4:00 PM ET
    command: python scripts/cron_predict_new.py
`

---

## Troubleshooting

### Migration errors

**Error:** `UndefinedColumn: column predictions.model_version_id does not exist`

**Fix:** `python scripts/migrate_db.py`

### No model versions

**Error:** `No model version found. Train and register a model first.`

**Fix:** 
`ash
python scripts/register_model.py --version v2026-01-02 --model-type RandomForest --artifact-path saved_models/ford_artifacts.joblib
`

### PyTorch models in cron

**Issue:** Cron predictor now supports both sklearn and PyTorch models (as of Jan 2, 2026).
- If errors occur, check artifacts: `python scripts/inspect_artifacts.py`

### Database connection

**Error:** `psycopg2.OperationalError: could not connect to server`

**Fix:** Verify `backend/database.ini` credentials and PostgreSQL is running.

### Insufficient predictions

**Issue:** Only 252 instead of 365 predictions

**Fix:** Download more historical data: `python scripts/download_data.py`

Check data per ticker: `SELECT ticker, COUNT(*) FROM stock_prices GROUP BY ticker;`

### CORS errors

**Fix:** Add CORS middleware in `backend/app.py`:
`python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
`

---

## Summary

| Task | Command | When |
|------|---------|------|
| Setup | `pip install -r requirements.txt && python scripts/migrate_db.py` | Once |
| Download data | `python scripts/download_data.py` | Periodically |
| Train models | `python stock_models/*.py` | Weekly/monthly |
| Register versions | `python scripts/batch_register.py --date-tag 2026-01-02` | After training |
| Generate predictions | `python scripts/cron_predict_new.py` | Daily at 4 PM ET |
| Update data | `python scripts/update_data.py` | Daily at 9:30 AM ET |
| Start dev server | `uvicorn backend.app:app --reload` | Development |
| Deploy | Push to main branch | As needed |


"""
Building the FastAPI backend for stock prediction models.
This module sets up the FastAPI app and defines endpoints for predictions, 
employing the best rules and practices for production readiness and api design.

Returns:
    json: Prediction results in JSON format
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException, Query
from typing import Optional
from datetime import datetime

from backend.database_service import load_predictions_for_version, get_latest_model_version


app = FastAPI()


@app.get("/")
async def root():
    return {"status": "ok", "service": "stock-predictions"}


@app.get("/predictions/{ticker}")
async def get_predictions(
    ticker: str,
    model_version_id: Optional[int] = Query(None, description="Model version ID; if omitted, uses latest version"),
):
    """
    Retrieve all 365 stored predictions for a ticker.
    
    Returns JSON array of prediction points (date + price) ready for frontend plotting.
    Omit model_version_id to get latest version predictions.
    
    Response:
    {
        "ticker": "TSLA",
        "model_version_id": 5,
        "predictions": [
            {"target_date": "2026-01-05", "predicted_price": 245.32, "trading_day_offset": 1},
            {"target_date": "2026-01-06", "predicted_price": 246.15, "trading_day_offset": 2},
            ...
        ]
    }
    """
    # If no version specified, get the latest
    if model_version_id is None:
        version = get_latest_model_version()
        if not version:
            raise HTTPException(status_code=404, detail="No model version available")
        model_version_id = version.id
    
    # Load all predictions for this ticker and version
    df = load_predictions_for_version(ticker.upper(), model_version_id)
    
    if df.empty:
        return {
            "ticker": ticker.upper(),
            "model_version_id": model_version_id,
            "predictions": []
        }
    
    # Convert to JSON-serializable list
    predictions = []
    for _, row in df.iterrows():
        predictions.append({
            "target_date": row['target_date'].isoformat(),
            "predicted_price": float(row['predicted_price']),
            "trading_day_offset": int(row['trading_day_offset']),
        })
    
    return {
        "ticker": ticker.upper(),
        "model_version_id": model_version_id,
        "predictions": predictions
    }

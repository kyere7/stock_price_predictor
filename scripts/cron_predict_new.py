"""
Cron prediction script (production version).

Generates 365 trading-day forecasts for all tickers using the latest model,
stores them with model versioning in PostgreSQL.

Run daily after market close (e.g., 22:00 UTC).
"""
import os
import sys

# Ensure project root is on sys.path so relative imports work when run as script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
# Ensure `scripts` directory is importable as top-level modules
SCRIPTS_DIR = os.path.join(ROOT, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from datetime import datetime
import traceback
import pandas as pd
import logging
import numpy as np
import torch

from production_utils import load_model_artifacts
from technical_indicators import add_technical_indicators
from backend.database_service import (
    load_all_tickers, 
    load_stock_data, 
    save_predictions_for_version,
    get_latest_model_version,
)
from backend.db_engine import create_engine_from_config, SessionLocal, init_db
from backend.market_calendar import get_next_n_trading_days

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HORIZONS = 365  # Store predictions for 365 trading days
SEQUENCE_WINDOW = 30  # HCLA models use 30-day sequence window


def ensure_db_initialized():
    engine = create_engine_from_config()
    SessionLocal.init(engine)
    init_db(engine)
    return engine


def iterative_forecast_for_ticker(ticker: str, model_version_id: int, num_days: int = HORIZONS) -> list:
    """
    Produce iterative forecasts for 365 trading days for one ticker.
    Supports both sklearn models (.predict) and PyTorch models (HybridCNN_LSTM_Attention).
    
    Returns list of tuples: [(target_date, trading_day_offset), predicted_price, ...]
    """
    ticker = ticker.upper()
    
    # Load model artifact
    try:
        artifacts = load_model_artifacts(ticker)
    except Exception as e:
        logger.warning(f"Skipping {ticker}: could not load artifacts: {e}")
        return [], []
    
    # Load recent historical data
    df = load_stock_data(ticker)
    if df.empty:
        logger.warning(f"No historical data for {ticker}, skipping")
        return [], []
    
    # Limit history for efficiency
    if len(df) > 365:
        df = df.tail(365)
    
    # Start iterative forecasting
    temp = df.copy()
    predictions = []
    offsets = []
    run_date = datetime.utcnow()
    
    # Get the next 365 trading days from today
    last_date = pd.Timestamp(df.index[-1]).date()
    future_trading_days = get_next_n_trading_days(last_date, num_days)
    
    if len(future_trading_days) < num_days:
        logger.warning(f"{ticker}: only {len(future_trading_days)} trading days available, expected {num_days}")
    
    model = artifacts.model
    scaler_X = artifacts.scaler_X
    scaler_y = artifacts.scaler_y
    
    # Detect model type
    is_sklearn = hasattr(model, 'predict')
    is_torch = hasattr(model, 'forward')  # PyTorch nn.Module has forward method
    
    if not is_sklearn and not is_torch:
        logger.warning(f"Skipping {ticker}: unknown model type (no .predict or .forward)")
        return [], []
    
    device = torch.device('cpu')  # CPU inference for cron jobs
    if is_torch:
        model.eval()
        model.to(device)
    
    for day_idx, target_date in enumerate(future_trading_days):
        offset = day_idx + 1  # 1-based offset
        
        try:
            # Compute indicators on current history
            feat_df = add_technical_indicators(temp)
            if feat_df.empty:
                logger.warning(f"{ticker}: no features at day {offset}, stopping")
                break
            
            # Get feature columns (exclude price columns)
            feature_cols = [c for c in feat_df.columns if c not in ['Open', 'High', 'Low', 'Volume', 'Adj Close']]
            if not feature_cols:
                logger.warning(f"{ticker}: no feature columns, stopping")
                break
            
            features = feat_df[feature_cols].values
            
            # Determine if we have enough history for sequence-based models
            if is_torch and len(features) < SEQUENCE_WINDOW:
                logger.warning(f"{ticker}: insufficient history ({len(features)} < {SEQUENCE_WINDOW}) for sequence model at day {offset}")
                break
            
            if is_sklearn:
                # Sklearn: use last row only
                X_scaled = scaler_X.transform(features[-1:])
                y_scaled = model.predict(X_scaled)
                y_arr = y_scaled.reshape(-1, 1)
                y_pred = float(scaler_y.inverse_transform(y_arr)[0, 0])
                
            else:  # PyTorch
                # Torch: use last SEQUENCE_WINDOW rows as input sequence
                X_seq = features[-SEQUENCE_WINDOW:]  # (sequence_window, num_features)
                X_scaled = scaler_X.transform(X_seq)  # Scale each row
                
                # Convert to tensor: (1, sequence_window, num_features)
                X_tensor = torch.from_numpy(X_scaled).float().unsqueeze(0).to(device)
                
                # Forward pass
                with torch.no_grad():
                    y_tensor = model(X_tensor)  # (1, 1)
                
                y_scaled = y_tensor.cpu().numpy()
                y_arr = y_scaled.reshape(-1, 1)
                y_pred = float(scaler_y.inverse_transform(y_arr)[0, 0])
            
            # Store prediction
            predictions.append(y_pred)
            offsets.append((target_date, offset))
            
            # Append predicted row to temp for next iteration
            new_row = {
                'Open': y_pred,
                'High': y_pred,
                'Low': y_pred,
                'Adj Close': y_pred,
                'Volume': 0,
            }
            temp.loc[pd.Timestamp(target_date)] = new_row
            
        except Exception as e:
            logger.error(f"{ticker}: prediction failed at day {offset}: {e}")
            traceback.print_exc()
            break
    
    logger.info(f"{ticker}: produced {len(predictions)} predictions")
    return offsets, predictions


def main():
    """Main cron job entry point."""
    engine = ensure_db_initialized()
    
    # Get or create model version
    # For now, we assume a model already exists; in production,
    # this would be triggered after model retraining.
    version = get_latest_model_version()
    if not version:
        logger.error("No model version found. Train and register a model first.")
        return
    
    logger.info(f"Using model version: {version.version_tag} (id={version.id})")
    
    # Load all tickers
    tickers = load_all_tickers()
    if not tickers:
        logger.info("No tickers in database")
        return
    
    logger.info(f"Forecasting for {len(tickers)} tickers")
    
    # Forecast for each ticker
    total_predictions = 0
    for ticker in tickers:
        try:
            offsets_dates, predictions = iterative_forecast_for_ticker(ticker, version.id)
            if predictions:
                saved = save_predictions_for_version(
                    ticker, 
                    offsets_dates, 
                    predictions, 
                    version.id,
                    datetime.utcnow()
                )
                total_predictions += saved
        except Exception as e:
            logger.error(f"Failed to forecast {ticker}: {e}")
            traceback.print_exc()
    
    logger.info(f"Cron job complete. Saved {total_predictions} total predictions.")


if __name__ == '__main__':
    main()

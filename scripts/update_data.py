"""
Update DB with latest stock data.
Intended to run as a cron job (e.g., on Render).

Behavior:
- For each ticker in DB (or default list), download data from last stored date+1 to today.
- Save new rows to PostgreSQL via existing database_service functions.

Note: Prediction responsibility is handled by `scripts/cron_predict.py`.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date, datetime, timedelta
import logging
import os
import pandas as pd

from backend.db_engine import create_engine_from_config, SessionLocal
from backend.database_service import (
    get_ticker_date_range,
    load_all_tickers,
)
from download_data import download_data, preprocess_data, save_stock_data_to_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ['F', 'CMCSA', 'PFE', 'FOLD', 'WBD', 'T', 'INTC', 'DNN', 'AAL', 'QUBT', 'ROIV', 'TSLA']


def ensure_db_initialized():
    engine = create_engine_from_config()
    SessionLocal.init(engine)
    return engine


# def next_day_predictions(ticker: str, artifacts_path=None):
#     """Load latest features for ticker, predict next-day price using saved artifact."""
#     try:
#         artifacts = load_model_artifacts(ticker)
#     except FileNotFoundError:
#         logger.warning("No artifact found for %s, skipping prediction", ticker)
#         return None

#     # Load full data to compute indicators
#     df = load_stock_data(ticker)
#     if df.empty:
#         logger.warning("No data available for %s to compute features", ticker)
#         return None

#     # Compute technical indicators (in-place returns new DF)
#     df = add_technical_indicators(df)

#     # Prepare feature row: drop price columns, keep last row
#     try:
#         feature_df = df.drop(columns=['Open', 'High', 'Low', 'Volume', 'Adj Close'])
#     except Exception:
#         logger.exception("Unexpected columns in dataframe for %s", ticker)
#         return None

#     last_row = feature_df.tail(1)
#     if last_row.empty:
#         logger.warning("No feature row for %s", ticker)
#         return None

#     # Scale
#     scaler_X = artifacts.scaler_X
#     scaler_y = artifacts.scaler_y

#     X_scaled = scaler_X.transform(last_row.values)

#     model = artifacts.model
#     # Predict: support sklearn (predict) and torch models
#     if hasattr(model, 'predict'):
#         y_scaled = model.predict(X_scaled).reshape(-1, 1)
#     else:
#         # assume torch model
#         import torch
#         X_tensor = torch.from_numpy(X_scaled).float()
#         model.eval()
#         with torch.no_grad():
#             y_scaled = model(X_tensor).cpu().numpy()
#         if y_scaled.ndim == 1:
#             y_scaled = y_scaled.reshape(-1, 1)

#     # Inverse transform
#     y_pred = scaler_y.inverse_transform(y_scaled)
#     return float(y_pred.ravel()[0])


def update_all(tickers=None):
    engine = ensure_db_initialized()
    tickers = tickers or load_all_tickers() or DEFAULT_TICKERS

    today = date.today()
    today_str = today.strftime('%Y-%m-%d')

    for ticker in tickers:
        logger.info("Processing %s", ticker)
        try:
            start_date, end_date = get_ticker_date_range(ticker)
            if end_date is None:
                start = '2000-01-01'
            else:
                # start from next calendar day after last stored
                start_dt = end_date + timedelta(days=1)
                start = start_dt.strftime('%Y-%m-%d')

            # If start > today, nothing to do
            if datetime.strptime(start, '%Y-%m-%d').date() > today:
                logger.info("%s already up-to-date (last date %s)", ticker, end_date)
            else:
                logger.info("Downloading %s from %s to %s", ticker, start, today_str)
                df = download_data(ticker, start, today_str)
                if df.empty:
                    logger.info("No new data for %s", ticker)
                else:
                    df = preprocess_data(df)
                    save_stock_data_to_db(ticker, df)

            # Data updated for this ticker (no prediction performed here)
            logger.info("%s data up-to-date through %s", ticker, today_str)

        except Exception as e:
            logger.exception("Failed to update %s: %s", ticker, e)
            
    logger.info("Update run complete")


if __name__ == '__main__':
    update_all()

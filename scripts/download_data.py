"""
Author: Gyabaah Kyere
Date: 2025-12-21
Description: 
Download stock data from yfinance and save to PostgreSQL database.
Data can be read from database instead of CSV files for ML models.
"""

import sys
from pathlib import Path

# Add parent directory to path to import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import os
import logging

# Other libraries
import yfinance as yf

# Backend imports
from backend.db_engine import SessionLocal, create_engine_from_config, init_db
from backend.database_service import save_stock_data as save_to_db
from backend.database_service import load_stock_data as load_from_db

# Logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Data collection
def download_data(tickers, start_date, end_date) -> pd.DataFrame:
    cols = ["Open", "High", "Low", "Adj Close", "Volume"]
    try:
        data = yf.download(tickers,
                           start=start_date,
                           end=end_date,
                           interval='1d',
                           auto_adjust=False,
                           progress=False)
    except Exception as e:
        logger.warning("%s: %s", tickers, e)
        return pd.DataFrame(columns=cols)

    if data is None or data.empty:
        logger.warning("%s: No data returned (may be delisted or invalid)", tickers)
        return pd.DataFrame(columns=cols)

    # Select expected columns safely
    try:
        data = data.loc[:, cols].copy()
    except Exception:
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data.xs(tickers, axis=1, level=1)[cols].copy()
            except Exception as e:
                logger.warning("%s: Failed to select OHLCV columns: %s", tickers, e)
                return pd.DataFrame(columns=cols)
        else:
            logger.warning("%s: Missing expected columns", tickers)
            return pd.DataFrame(columns=cols)

    data.dropna(inplace=True)
    return data

def save_stock_data(df: pd.DataFrame, filename: str, output_dir: str = 'stock_data') -> None:
    """Save stock data to CSV in a specified directory (legacy, kept for backwards compatibility)."""
    if df is None or df.empty:
        logger.info("%s: No data to save, skipping file %s", filename, output_dir)
        return
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath)


def save_stock_data_to_db(ticker: str, df: pd.DataFrame) -> None:
    """Save stock data to PostgreSQL database."""
    if df is None or df.empty:
        logger.warning(f"No data to save for {ticker}")
        return
    
    try:
        session = SessionLocal.get_session()
        save_to_db(ticker, df, session=session)
        session.close()
    except Exception as e:
        logger.error(f"Failed to save {ticker} to database: {e}")


# read data and preprocess
def read_stock_data_from_db(ticker: str) -> pd.DataFrame:
    """Read stock data from PostgreSQL database."""
    try:
        session = SessionLocal.get_session()
        df = load_from_db(ticker, session=session)
        session.close()
        return df
    except Exception as e:
        logger.error(f"Failed to load {ticker} from database: {e}")
        return pd.DataFrame()


def read_stock_data(filepath: str):
    """Read stock data from a CSV file (legacy, kept for backwards compatibility)."""
    return pd.read_csv(filepath, parse_dates=True, index_col=0)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess stock data.
    Ensures numeric types and removes NaN values.
    """
    # Ensure all data columns are numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows that resulted in NaN values after numeric conversion
    df.dropna(inplace=True)
    
    return df

def main():
    """Download stock data and save to PostgreSQL database."""
    tickers = ['F', 'CMCSA', 'PFE', 'FOLD',
               'WBD', 'T', 'INTC', 'DNN',
               'AAL', 'QUBT', 'ROIV', 'TSLA']
    
    start_date = '2000-01-01'
    end_date = '2025-01-01'

    # Initialize database engine and create tables
    engine = create_engine_from_config()
    SessionLocal.init(engine)
    init_db(engine)

    print(f"\nDownloading stock data for {len(tickers)} tickers...")
    print(f"Date range: {start_date} to {end_date}\n")

    success_count = 0
    failed_tickers = []

    for ticker in tickers:
        try:
            logger.info(f"Downloading {ticker}...")
            data = download_data(ticker, start_date, end_date)
            
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                failed_tickers.append(ticker)
                continue
            
            # Preprocess the data
            data = preprocess_data(data)
            
            # Save to database
            save_stock_data_to_db(ticker, data)
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
            failed_tickers.append(ticker)

    # Print summary
    print("\n" + "="*60)
    print(f"✓ Successfully downloaded and saved: {success_count}/{len(tickers)} tickers")
    if failed_tickers:
        print(f"✗ Failed tickers: {', '.join(failed_tickers)}")
    print("="*60)


if __name__ == "__main__":
    main()
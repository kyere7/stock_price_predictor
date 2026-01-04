"""
Database service functions for saving and loading stock data.
Provides high-level interface for data operations.
"""
from datetime import datetime
from typing import List, Optional
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from backend.db_engine import SessionLocal
from backend.models import StockPrice, Prediction, ModelVersion


def save_stock_data(ticker: str, df: pd.DataFrame, session: Optional[Session] = None) -> int:
    """
    Save stock price data to database.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'F', 'TSLA')
        df: DataFrame with columns [Open, High, Low, Adj Close, Volume]
            Index must be dates (pd.DatetimeIndex)
        session: Database session. If None, creates new session
        
    Returns:
        Number of rows inserted/updated
    """
    close_session = False
    if session is None:
        session = SessionLocal.get_session()
        close_session = True
    
    try:
        count = 0
        for date, row in df.iterrows():
            # Convert index to date if datetime
            date = pd.Timestamp(date).date()
            
            # Check if record exists
            existing = session.query(StockPrice).filter(
                and_(
                    StockPrice.ticker == ticker,
                    StockPrice.date == date
                )
            ).first()
            
            if existing:
                # Update existing record
                existing.open = float(row.get('Open', 0))
                existing.high = float(row.get('High', 0))
                existing.low = float(row.get('Low', 0))
                existing.adj_close = float(row.get('Adj Close', 0))
                existing.volume = int(row.get('Volume', 0))
            else:
                # Create new record
                stock = StockPrice(
                    ticker=ticker,
                    date=date,
                    open=float(row.get('Open', 0)),
                    high=float(row.get('High', 0)),
                    low=float(row.get('Low', 0)),
                    adj_close=float(row.get('Adj Close', 0)),
                    volume=int(row.get('Volume', 0)),
                )
                session.add(stock)
            
            count += 1
        
        session.commit()
        print(f"✓ Saved {count} records for {ticker}")
        return count
        
    except Exception as e:
        session.rollback()
        print(f"✗ Error saving {ticker}: {e}")
        raise
    finally:
        if close_session:
            session.close()


def load_stock_data(
    ticker: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    session: Optional[Session] = None,
) -> pd.DataFrame:
    """
    Load stock price data from database.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Filter from this date (optional)
        end_date: Filter to this date (optional)
        session: Database session. If None, creates new session
        
    Returns:
        DataFrame with columns [Open, High, Low, Adj Close, Volume]
        Index is Date
    """
    close_session = False
    if session is None:
        session = SessionLocal.get_session()
        close_session = True
    
    try:
        query = session.query(StockPrice).filter(StockPrice.ticker == ticker)
        
        if start_date:
            query = query.filter(StockPrice.date >= start_date)
        if end_date:
            query = query.filter(StockPrice.date <= end_date)
        
        query = query.order_by(StockPrice.date)
        results = query.all()
        
        if not results:
            print(f"No data found for {ticker}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = {
            'Open': [r.open for r in results],
            'High': [r.high for r in results],
            'Low': [r.low for r in results],
            'Adj Close': [r.adj_close for r in results],
            'Volume': [r.volume for r in results],
        }
        df = pd.DataFrame(data, index=[r.date for r in results])
        df.index.name = 'Date'
        
        print(f"✓ Loaded {len(df)} records for {ticker}")
        return df
        
    finally:
        if close_session:
            session.close()


def load_all_tickers(session: Optional[Session] = None) -> List[str]:
    """
    Get list of all tickers in the database.
    
    Args:
        session: Database session. If None, creates new session
        
    Returns:
        List of ticker symbols
    """
    close_session = False
    if session is None:
        session = SessionLocal.get_session()
        close_session = True
    
    try:
        tickers = session.query(StockPrice.ticker).distinct().all()
        return [t[0] for t in tickers]
    finally:
        if close_session:
            session.close()


def delete_ticker_data(ticker: str, session: Optional[Session] = None) -> int:
    """
    Delete all data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        session: Database session. If None, creates new session
        
    Returns:
        Number of rows deleted
    """
    close_session = False
    if session is None:
        session = SessionLocal.get_session()
        close_session = True
    
    try:
        count = session.query(StockPrice).filter(StockPrice.ticker == ticker).delete()
        session.commit()
        print(f"✓ Deleted {count} records for {ticker}")
        return count
    finally:
        if close_session:
            session.close()


def get_ticker_date_range(ticker: str, session: Optional[Session] = None) -> tuple:
    """
    Get earliest and latest date for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        session: Database session. If None, creates new session
        
    Returns:
        Tuple of (start_date, end_date)
    """
    close_session = False
    if session is None:
        session = SessionLocal.get_session()
        close_session = True
    
    try:
        from sqlalchemy import func

        result = (
            session.query(
                func.min(StockPrice.date),
                func.max(StockPrice.date)
            )
            .filter(StockPrice.ticker == ticker)
            .first()
        )

        
        return result if result else (None, None)
    finally:
        if close_session:
            session.close()


def save_predictions(preds: list, session: Optional[Session] = None) -> int:
    """
    Save a list of prediction dicts to the database.

    Each dict should include keys: ticker, target_date (date), horizon_days (int), predicted_price (float), method (optional), run_date (optional)
    Returns number of rows inserted.
    """
    close_session = False
    if session is None:
        session = SessionLocal.get_session()
        close_session = True

    try:
        count = 0
        for p in preds:
            # Normalize
            ticker = p.get('ticker')
            target_date = p.get('target_date')
            horizon = int(p.get('horizon_days'))
            price = float(p.get('predicted_price'))
            method = p.get('method')
            run_date = p.get('run_date')

            # Check existing
            existing = session.query(Prediction).filter(
                Prediction.ticker == ticker,
                Prediction.target_date == target_date,
                Prediction.horizon_days == horizon
            ).first()

            if existing:
                existing.predicted_price = price
                existing.method = method
                existing.run_date = run_date or existing.run_date
            else:
                pred = Prediction(
                    ticker=ticker,
                    target_date=target_date,
                    horizon_days=horizon,
                    predicted_price=price,
                    method=method,
                    run_date=run_date,
                )
                session.add(pred)
            count += 1

        session.commit()
        print(f"✓ Saved {count} prediction rows")
        return count
    except Exception:
        session.rollback()
        raise
    finally:
        if close_session:
            session.close()


def load_predictions(ticker: str, horizon_days: Optional[int] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, session: Optional[Session] = None) -> pd.DataFrame:
    """
    Load predictions for a ticker, optionally filtering by horizon and date range.

    Returns a DataFrame with columns: ['ticker','target_date','horizon_days','predicted_price','method','run_date']
    """
    close_session = False
    if session is None:
        session = SessionLocal.get_session()
        close_session = True

    try:
        query = session.query(Prediction).filter(Prediction.ticker == ticker)
        if horizon_days is not None:
            query = query.filter(Prediction.horizon_days == int(horizon_days))
        if start_date is not None:
            query = query.filter(Prediction.target_date >= start_date)
        if end_date is not None:
            query = query.filter(Prediction.target_date <= end_date)

        query = query.order_by(Prediction.target_date)
        results = query.all()
        if not results:
            return pd.DataFrame()

        data = {
            'ticker': [r.ticker for r in results],
            'target_date': [r.target_date for r in results],
            'horizon_days': [r.horizon_days for r in results],
            'predicted_price': [r.predicted_price for r in results],
            'method': [r.method for r in results],
            'run_date': [r.run_date for r in results],
        }
        df = pd.DataFrame(data)
        return df
    finally:
        if close_session:
            session.close()


def create_model_version(version_tag: str, model_type: str, artifact_path: str, trained_at: datetime, notes: Optional[str] = None, session: Optional[Session] = None) -> ModelVersion:
    """
    Create a new model version record.
    
    Args:
        version_tag: Unique version identifier (e.g., v2026-01-01)
        model_type: Type of model (e.g., RandomForest, HCLA)
        artifact_path: Path to saved model artifact
        trained_at: Training completion datetime
        notes: Optional notes or metrics summary
        session: Database session. If None, creates new session
    
    Returns:
        Created ModelVersion instance
    """
    close_session = False
    if session is None:
        session = SessionLocal.get_session()
        close_session = True
    
    try:
        version = ModelVersion(
            version_tag=version_tag,
            model_type=model_type,
            artifact_path=artifact_path,
            trained_at=trained_at,
            is_active=1,
            notes=notes,
        )
        session.add(version)
        session.commit()
        print(f"✓ Created model version: {version_tag}")
        return version
    except Exception as e:
        session.rollback()
        print(f"✗ Error creating model version: {e}")
        raise
    finally:
        if close_session:
            session.close()


def get_latest_model_version(session: Optional[Session] = None) -> Optional[ModelVersion]:
    """
    Get the most recently trained model version.
    
    Args:
        session: Database session. If None, creates new session
    
    Returns:
        ModelVersion instance or None if no versions exist
    """
    close_session = False
    if session is None:
        session = SessionLocal.get_session()
        close_session = True
    
    try:
        version = session.query(ModelVersion).filter(ModelVersion.is_active == 1).order_by(desc(ModelVersion.trained_at)).first()
        return version
    finally:
        if close_session:
            session.close()


def save_predictions_for_version(ticker: str, target_dates_and_offsets: List[tuple], predicted_prices: List[float], model_version_id: int, run_date: datetime, session: Optional[Session] = None) -> int:
    """
    Save predictions for a specific model version in bulk.
    
    Args:
        ticker: Stock ticker symbol
        target_dates_and_offsets: List of tuples (target_date, trading_day_offset)
        predicted_prices: List of predicted prices (same length as target_dates_and_offsets)
        model_version_id: ID of the model version
        run_date: When this prediction batch was generated
        session: Database session. If None, creates new session
    
    Returns:
        Number of rows inserted/updated
    """
    close_session = False
    if session is None:
        session = SessionLocal.get_session()
        close_session = True
    
    try:
        count = 0
        for (target_date, offset), price in zip(target_dates_and_offsets, predicted_prices):
            # Upsert: update if exists, else create
            existing = session.query(Prediction).filter(
                Prediction.ticker == ticker,
                Prediction.model_version_id == model_version_id,
                Prediction.target_date == target_date
            ).first()
            
            if existing:
                existing.predicted_price = price
                existing.trading_day_offset = offset
                existing.run_date = run_date
            else:
                pred = Prediction(
                    ticker=ticker,
                    model_version_id=model_version_id,
                    target_date=target_date,
                    trading_day_offset=offset,
                    predicted_price=price,
                    run_date=run_date,
                )
                session.add(pred)
            count += 1
        
        session.commit()
        print(f"✓ Saved {count} predictions for {ticker} (version {model_version_id})")
        return count
    except Exception as e:
        session.rollback()
        print(f"✗ Error saving predictions: {e}")
        raise
    finally:
        if close_session:
            session.close()


def load_predictions_for_version(ticker: str, model_version_id: int, session: Optional[Session] = None) -> pd.DataFrame:
    """
    Load all predictions for a ticker and model version, sorted by target_date.
    
    Args:
        ticker: Stock ticker symbol
        model_version_id: ID of the model version
        session: Database session. If None, creates new session
    
    Returns:
        DataFrame with columns [ticker, target_date, trading_day_offset, predicted_price, run_date, model_version_id]
    """
    close_session = False
    if session is None:
        session = SessionLocal.get_session()
        close_session = True
    
    try:
        results = session.query(Prediction).filter(
            Prediction.ticker == ticker,
            Prediction.model_version_id == model_version_id
        ).order_by(Prediction.target_date).all()
        
        if not results:
            return pd.DataFrame()
        
        data = {
            'ticker': [r.ticker for r in results],
            'target_date': [r.target_date for r in results],
            'trading_day_offset': [r.trading_day_offset for r in results],
            'predicted_price': [r.predicted_price for r in results],
            'run_date': [r.run_date for r in results],
            'model_version_id': [r.model_version_id for r in results],
        }
        return pd.DataFrame(data)
    finally:
        if close_session:
            session.close()

"""
SQLAlchemy ORM models for stock price data.
Defines the database schema for storing stock market data.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Date, Index, UniqueConstraint, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class StockPrice(Base):
    """
    ORM model for storing stock price data.
    Unified table for all tickers - use 'ticker' column to filter by stock.
    """
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)  # F, TSLA, INTC, etc.
    date = Column(Date, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    adj_close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Ensure unique entries per ticker per date (no duplicates)
    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='uq_ticker_date'),
        Index('ix_ticker_date', 'ticker', 'date'),
    )
    
    def __repr__(self):
        return f"<StockPrice(ticker={self.ticker}, date={self.date}, adj_close={self.adj_close})>"


class ModelVersion(Base):
    """
    ORM model for storing trained model metadata and versioning.
    Each time a model is retrained, a new row is created.
    """
    __tablename__ = 'model_versions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_tag = Column(String(100), nullable=False, unique=True)  # e.g., v2026-01-01
    model_type = Column(String(50), nullable=True)  # e.g., "RandomForest", "HCLA"
    artifact_path = Column(String(500), nullable=True)  # Path to saved model
    trained_at = Column(DateTime, nullable=False)
    is_active = Column(Integer, default=1)  # 1 = active, 0 = archived
    notes = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ModelVersion(id={self.id}, tag={self.version_tag}, trained_at={self.trained_at})>"


class Prediction(Base):
    """
    ORM model for storing forecasted prices for tickers.
    Each row represents a prediction for a specific target date,
    produced by a specific model version on a specific run_date.
    """
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    model_version_id = Column(Integer, ForeignKey('model_versions.id'), nullable=False, index=True)
    run_date = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    target_date = Column(Date, nullable=False, index=True)
    trading_day_offset = Column(Integer, nullable=False)  # 1 = next trading day, ..., 365
    predicted_price = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('ticker', 'model_version_id', 'target_date', name='uq_pred_ticker_version_target'),
        Index('ix_pred_ticker_version_offset', 'ticker', 'model_version_id', 'trading_day_offset'),
        Index('ix_pred_ticker_target', 'ticker', 'target_date'),
    )

    def __repr__(self):
        return f"<Prediction(ticker={self.ticker}, target_date={self.target_date}, offset={self.trading_day_offset}, price={self.predicted_price})>"
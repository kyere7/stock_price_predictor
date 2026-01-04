"""
Database engine and session management.
Handles connection pooling, session creation, and engine initialization.
"""
import os
from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from backend.config import load_config
from backend.models import Base


def create_engine_from_config(
    config_file='C:/Users/Gyabaah/Desktop/stock_predictor/backend/database.ini',
    section='postgresql',
    echo=False,
    pool_size=10,
    max_overflow=20,
):
    """
    Create SQLAlchemy engine from database.ini config file.
    
    Args:
        config_file: Path to database.ini
        section: Section name in config file (default: 'postgresql')
        echo: If True, log all SQL statements
        pool_size: Number of connections to keep in pool
        max_overflow: Additional connections beyond pool_size
        
    Returns:
        SQLAlchemy Engine
    """
    cfg = load_config(config_file, section=section)
    
    url = URL.create(
        "postgresql+psycopg2",
        username=cfg.get("user"),
        password=cfg.get("password"),
        host=cfg.get("host"),
        port=int(cfg.get("port")) if cfg.get("port") else 5432,
        database=cfg.get("database"),
    )
    
    engine = create_engine(
        url,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,  # Verify connections before using
    )
    
    return engine


def create_engine_from_url(database_url=None, echo=False):
    """
    Create SQLAlchemy engine from DATABASE_URL environment variable or string.
    
    Args:
        database_url: Full database URL string. If None, reads from DATABASE_URL env var
        echo: If True, log all SQL statements
        
    Returns:
        SQLAlchemy Engine
    """
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not provided and not set in environment")
    
    engine = create_engine(
        url,
        echo=echo,
        pool_pre_ping=True,
    )
    
    return engine


def init_db(engine):
    """
    Create all tables defined in models.
    
    Args:
        engine: SQLAlchemy Engine
    """
    Base.metadata.create_all(engine)
    print("✓ Database tables created/verified")


class SessionLocal:
    """Session factory for database connections."""
    _session_factory = None
    _engine = None
    
    @classmethod
    def init(cls, engine):
        """Initialize session factory with an engine."""
        cls._engine = engine
        cls._session_factory = sessionmaker(
            bind=engine,
            expire_on_commit=False,
            autoflush=False,
        )
        init_db(engine)
    
    @classmethod
    def get_session(cls) -> Session:
        """Get a new database session."""
        if cls._session_factory is None:
            raise RuntimeError("SessionLocal not initialized. Call SessionLocal.init(engine) first.")
        return cls._session_factory()
    
    @classmethod
    def get_engine(cls):
        """Get the engine instance."""
        if cls._engine is None:
            raise RuntimeError("SessionLocal not initialized. Call SessionLocal.init(engine) first.")
        return cls._engine


# Initialize on import (development mode)
# For production, call SessionLocal.init(engine) explicitly in your app startup
if __name__ != "__main__":
    try:
        _engine = create_engine_from_config()
        SessionLocal.init(_engine)
    except Exception as e:
        print(f"Warning: Could not auto-initialize database engine: {e}")

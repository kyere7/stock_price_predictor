import sys
from pathlib import Path

# Add parent directory to path to import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from backend.config import load_config  # or from config import load_config

def create_engine_from_config(filename=None, section='postgresql', **kw):
    cfg = load_config(filename)  # returns dict: {'host':..,'user':..,'password':..,'database':..,'port':..}
    url = URL.create(
        "postgresql+psycopg2",
        username=cfg.get("user"),
        password=cfg.get("password"),
        host=cfg.get("host"),
        port=int(cfg.get("port")) if cfg.get("port") else None,
        database=cfg.get("database"),
        query={}  # add sslmode etc if needed
    )
    engine = create_engine(url, pool_pre_ping=True, **kw)
    return engine

engine = create_engine_from_config('C:/Users/Gyabaah/Desktop/stock_predictor/backend/database.ini')
with engine.connect() as conn:
    print("Database connection successful.")
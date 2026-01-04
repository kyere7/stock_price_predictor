from configparser import ConfigParser


parser = ConfigParser()
import os
from configparser import ConfigParser
from urllib.parse import urlparse, unquote

def _parse_database_url(db_url: str) -> dict:
    parsed = urlparse(db_url)
    user = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None
    host = parsed.hostname
    port = str(parsed.port) if parsed.port else None
    database = parsed.path.lstrip('/') if parsed.path else None
    return {
        'user': user,
        'password': password,
        'host': host,
        'port': port,
        'database': database,
    }

def load_config(filename=None, section='postgresql'):
    """Load DB config.

    Priority:
    1. `DATABASE_URL` environment variable (preferred on Render/NEON)
    2. INI file at `filename` or `backend/database.ini` (fallback)
    Returns a dict with keys: user, password, host, port, database
    """
    # 1) DATABASE_URL (e.g. postgresql://user:pass@host:port/dbname)
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        return _parse_database_url(db_url)

    # 2) Fallback to ini file
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), 'database.ini')

    parser = ConfigParser()
    read_files = parser.read(filename)
    if not read_files:
        raise FileNotFoundError(f"Database config file not found: {filename}")

    if parser.has_section(section):
        params = parser.items(section)
        config = {param[0]: param[1] for param in params}
        return config

    raise Exception(f"Section {section} not found in the {filename} file")

if __name__ == '__main__':
    try:
        cfg = load_config()
        print(cfg)
    except Exception as e:
        print('Error loading config:', e)


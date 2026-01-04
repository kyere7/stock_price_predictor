"""
Batch register all trained models in saved_models/ directory.

Creates ticker-specific version tags for each model (e.g., f_v1_2026-01-01, tsla_v1_2026-01-01).

Usage:
    python scripts/batch_register.py [--date-tag 2026-01-01]
    
If --date-tag is not provided, uses current date (2026-01-01).
Ticker will be automatically extracted from artifact filenames.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import argparse
import joblib
from datetime import datetime
from backend.db_engine import create_engine_from_config, SessionLocal, init_db
from backend.database_service import create_model_version

# Map model class name to readable type
MODEL_TYPE_MAP = {
    'RandomForestRegressor': 'RandomForest',
    'HybridCNN_LSTM_Attention': 'HybridCNN_LSTM_Attention',
}


def get_ticker_from_filename(filename):
    """Extract ticker from artifact filename (e.g., 'ford_artifacts.joblib' → 'F')."""
    # Remove '_artifacts.joblib' suffix
    name = filename.replace('_artifacts.joblib', '')
    
    # Map common artifact names to tickers
    ticker_map = {
        'ford': 'F',
        'cmcsa': 'CMCSA',
        'fold': 'FOLD',
        'intc': 'INTC',
        'pfe': 'PFE',
        'roiv': 'ROIV',
        'tsla': 'TSLA',
        't': 'T',
        'dnn': 'DNN',
        'aal': 'AAL',
        'qubt': 'QUBT',
        'wbd': 'WBD',
    }
    
    return ticker_map.get(name.lower(), name.upper())


def get_model_type(artifact_dict):
    """Infer model type from loaded artifact dict."""
    model = artifact_dict.get('model')
    if model is None:
        return 'Unknown'
    
    class_name = model.__class__.__name__
    return MODEL_TYPE_MAP.get(class_name, class_name)


def batch_register(date_tag, models_dir='saved_models'):
    """Register all artifacts in models_dir with ticker-specific version tags."""
    # Initialize DB
    engine = create_engine_from_config()
    SessionLocal.init(engine)
    init_db(engine)
    
    # Find all artifact files
    artifacts = []
    if os.path.isdir(models_dir):
        for fn in sorted(os.listdir(models_dir)):
            if fn.endswith('_artifacts.joblib'):
                artifacts.append((fn, os.path.join(models_dir, fn)))
    
    if not artifacts:
        print(f"No artifacts found in {models_dir}")
        return
    
    print(f"Found {len(artifacts)} artifact(s). Registering with date tag: {date_tag}\n")
    
    registered = 0
    failed = []
    
    for filename, filepath in artifacts:
        try:
            # Extract ticker from filename
            ticker = get_ticker_from_filename(filename)
            
            # Create ticker-specific version tag
            version_tag = f"{ticker.lower()}_v1_{date_tag}"
            
            # Load artifact to determine model type
            artifact_dict = joblib.load(filepath)
            model_type = get_model_type(artifact_dict)
            
            # Create version
            version = create_model_version(
                version_tag=version_tag,
                model_type=model_type,
                artifact_path=filepath,
                trained_at=datetime.utcnow(),
                notes=f"Artifact: {filename}",
            )
            print(f"✓ {filename:40} → {version_tag:30} {model_type:20} (id={version.id})")
            registered += 1
            
        except Exception as e:
            print(f"✗ {filename:40} → ERROR: {e}")
            failed.append((filename, str(e)))
    
    print(f"\n{'='*80}")
    print(f"Registered: {registered}/{len(artifacts)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for fn, err in failed:
            print(f"  - {fn}: {err}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Batch register all model artifacts with ticker-specific versions")
    parser.add_argument("--date-tag", required=False, help="Date tag for version (e.g., 2026-01-15)")
    parser.add_argument("--models-dir", default="C:\\Users\\Gyabaah\\Desktop\\stock_predictor\\saved_models", help="Directory containing artifacts")
    
    args = parser.parse_args()
    
    # If date-tag not provided, prompt for it
    date_tag = args.date_tag
    if not date_tag:
        # Suggest a default based on today's date
        default_date = datetime.now().strftime('%Y-%m-%d')
        date_tag = input(f"Enter date tag [{default_date}]: ").strip() or default_date
    
    print(f"Using date tag: {date_tag}\n")
    batch_register(date_tag, args.models_dir)


if __name__ == '__main__':
    main()

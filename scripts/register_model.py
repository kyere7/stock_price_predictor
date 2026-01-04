"""
Register a trained model version in the database.

Use this script after training/retraining a model to create a ModelVersion record
and make it available for the cron predictor to use.

Example usage:
    python scripts/register_model.py --version v2026-01-01 --model-type RandomForest --artifact-path saved_models/ford_artifacts.joblib
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime
from backend.db_engine import create_engine_from_config, SessionLocal, init_db
from backend.database_service import create_model_version


def main():
    parser = argparse.ArgumentParser(description="Register a trained model version")
    parser.add_argument("--version", required=True, help="Version tag (e.g., v2026-01-01)")
    parser.add_argument("--model-type", required=True, help="Model type (e.g., RandomForest, HCLA)")
    parser.add_argument("--artifact-path", required=True, help="Path to model artifact")
    parser.add_argument("--notes", default=None, help="Optional notes")
    
    args = parser.parse_args()
    
    # Initialize DB
    engine = create_engine_from_config()
    SessionLocal.init(engine)
    init_db(engine)
    
    # Create version
    version = create_model_version(
        version_tag=args.version,
        model_type=args.model_type,
        artifact_path=args.artifact_path,
        trained_at=datetime.utcnow(),
        notes=args.notes,
    )
    
    print(f"✓ Registered model version: {version.version_tag} (id={version.id})")


if __name__ == '__main__':
    main()

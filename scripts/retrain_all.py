"""
Retrain all models and automatically register them as new model versions.

This script:
1. Trains/retrains all models in stock_models/
2. Saves artifacts to saved_models/<ticker>_artifacts.joblib
3. Automatically registers them with ticker-specific version tags
   (e.g., f_v1_2026-01-15, tsla_v1_2026-01-15, etc.)

The cron predictor (cron_predict_new.py) will automatically use the latest
registered version for each ticker to generate 365-day forecasts.

Usage:
    python scripts/retrain_all.py [--date-tag 2026-01-15]
    
If --date-tag not provided, uses current date.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import subprocess
import argparse
from datetime import datetime
import glob


def find_model_scripts(stock_models_dir='stock_models'):
    """Find all model training scripts (exclude __pycache__)."""
    scripts = []
    for fn in sorted(os.listdir(stock_models_dir)):
        if fn.endswith('_model.py') and fn != '__pycache__':
            scripts.append(os.path.join(stock_models_dir, fn))
    return scripts


def train_all_models(stock_models_dir='stock_models'):
    """Run all model training scripts sequentially."""
    scripts = find_model_scripts(stock_models_dir)
    
    if not scripts:
        print(f"No model scripts found in {stock_models_dir}")
        return False
    
    print(f"Found {len(scripts)} model(s) to train:\n")
    for script in scripts:
        print(f"  - {script}")
    
    print(f"\n{'='*80}")
    
    failed = []
    for script in scripts:
        print(f"\nTraining: {script}")
        try:
            result = subprocess.run(
                [sys.executable, script],
                cwd=os.path.dirname(os.path.abspath(__file__)).replace('scripts', ''),
                capture_output=False,
                timeout=3600  # 1 hour timeout per model
            )
            if result.returncode != 0:
                failed.append(script)
                print(f"✗ Failed with exit code {result.returncode}")
            else:
                print(f"✓ Completed successfully")
        except subprocess.TimeoutExpired:
            failed.append(script)
            print(f"✗ Training timed out (>1 hour)")
        except Exception as e:
            failed.append(script)
            print(f"✗ Error: {e}")
    
    print(f"\n{'='*80}")
    print(f"Training complete: {len(scripts) - len(failed)}/{len(scripts)} succeeded")
    if failed:
        print(f"Failed models:")
        for script in failed:
            print(f"  - {script}")
        return False
    
    return True


def register_models(date_tag, models_dir='saved_models'):
    """Register trained models via batch_register.py"""
    print(f"\nRegistering models with date tag: {date_tag}")
    
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/batch_register.py', '--date-tag', date_tag],
            capture_output=False,
            timeout=300
        )
        if result.returncode != 0:
            print(f"✗ Registration failed with exit code {result.returncode}")
            return False
        print(f"✓ Models registered")
        return True
    except Exception as e:
        print(f"✗ Error registering models: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Retrain all models and register with ticker-specific versions")
    parser.add_argument("--date-tag", required=False, help="Date tag for versions (e.g., 2026-01-15)")
    
    args = parser.parse_args()
    
    # Determine date tag
    date_tag = args.date_tag
    if not date_tag:
        date_tag = datetime.now().strftime('%Y-%m-%d')
        print(f"No date tag specified. Using: {date_tag}")
    
    print(f"{'='*80}")
    print(f"RETRAIN ALL MODELS - Date Tag: {date_tag}")
    print(f"{'='*80}\n")
    
    # Step 1: Train all models
    print("Step 1: Training all models...")
    if not train_all_models():
        print("\n✗ Training failed. Aborting.")
        return 1
    
    # Step 2: Register models
    print("\nStep 2: Registering models...")
    if not register_models(date_tag):
        print("\n✗ Registration failed. Aborting.")
        return 1
    
    print(f"\n{'='*80}")
    print(f"✓ RETRAINING COMPLETE")
    print(f"All models trained and registered with ticker-specific versions:")
    print(f"  f_v1_{date_tag}, tsla_v1_{date_tag}, cmcsa_v1_{date_tag}, etc.")
    print(f"Cron predictor will automatically use these versions for forecasts.")
    print(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

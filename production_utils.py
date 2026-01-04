"""
Production utilities for saving, loading, and managing model artifacts.
Provides consistent interface for model + scaler persistence.
"""

import joblib
import os
import sys

import sys
from pathlib import Path

# Add parent directory to path to import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))


class ModelArtifacts:
    """Container for model and preprocessing artifacts needed for inference."""
    
    def __init__(self, model, scaler_X, scaler_y, ticker=None):
        """
        Initialize model artifacts.
        
        Args:
            model: Trained PyTorch model
            scaler_X: MinMaxScaler fitted on features
            scaler_y: MinMaxScaler fitted on target
            ticker: Stock ticker symbol (optional, for tracking)
        """
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.ticker = ticker
    
    def save(self, filepath):
        """
        Save all artifacts to a single joblib file.
        
        Args:
            filepath: Path where to save artifacts (e.g., 'saved_models/ford_artifacts.joblib')
        """
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        artifacts = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'ticker': self.ticker
        }
        
        joblib.dump(artifacts, filepath)
        print(f"✓ Artifacts saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """
        Load all artifacts from a joblib file.
        
        Args:
            filepath: Path to saved artifacts file
            
        Returns:
            ModelArtifacts instance
        """
        artifacts = joblib.load(filepath)
        return ModelArtifacts(
            model=artifacts['model'],
            scaler_X=artifacts['scaler_X'],
            scaler_y=artifacts['scaler_y'],
            ticker=artifacts.get('ticker')
        )


def save_model_artifacts(model, scaler_X, scaler_y, ticker, models_dir='saved_models'):
    """
    Convenience function to save model artifacts for a specific ticker.
    
    Args:
        model: Trained PyTorch model
        scaler_X: MinMaxScaler for features
        scaler_y: MinMaxScaler for target
        ticker: Stock ticker symbol
        models_dir: Directory to save to (default: 'saved_models')
    """
    artifacts = ModelArtifacts(model, scaler_X, scaler_y, ticker)
    filepath = os.path.join(models_dir, f'{ticker.lower()}_artifacts.joblib')
    artifacts.save(filepath)
    return filepath


def load_model_artifacts(ticker, models_dir='C:/Users/Gyabaah/Desktop/stock_predictor/saved_models'):
    """
    Convenience function to load model artifacts for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        models_dir: Directory where artifacts are saved (default: 'saved_models')
        
    Returns:
        ModelArtifacts instance
    """
    # Ensure project root is on sys.path so custom model classes (e.g., HCLA_model)
    # referenced by pickled artifacts can be imported during unpickling.
    project_root = os.path.abspath(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Primary candidate
    candidate = os.path.join(models_dir, f'{ticker.lower()}_artifacts.joblib')
    if os.path.exists(candidate):
        return ModelArtifacts.load(candidate)

    # Fallback: search for any artifact file containing the ticker string
    if os.path.isdir(models_dir):
        for fn in os.listdir(models_dir):
            if fn.endswith('_artifacts.joblib') and ticker.lower() in fn.lower():
                return ModelArtifacts.load(os.path.join(models_dir, fn))

    raise FileNotFoundError(f"No artifacts found for ticker '{ticker}' in {models_dir}")

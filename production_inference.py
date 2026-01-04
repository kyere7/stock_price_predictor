"""
Production inference template for using trained models.

Usage example:
    artifacts = load_model_artifacts('FORD')
    predictions = predict_stock_price(artifacts, new_feature_data)
"""

import torch
import numpy as np
from production_utils import load_model_artifacts


def predict_stock_price(artifacts, feature_data_scaled, device='cpu'):
    """
    Make predictions using a loaded model artifact.
    
    Args:
        artifacts: ModelArtifacts instance from load_model_artifacts()
        feature_data_scaled: Scaled feature data (already preprocessed with scaler_X)
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        predictions: Predictions in original stock price scale (not scaled)
    """
    model = artifacts.model
    scaler_y = artifacts.scaler_y
    
    model.eval()
    model.to(device)
    
    # Convert to tensor
    X_tensor = torch.from_numpy(feature_data_scaled).float().to(device)
    
    # Make predictions
    with torch.no_grad():
        y_preds_scaled = model(X_tensor).cpu().numpy()
    
    # Inverse transform to original scale
    y_preds = scaler_y.inverse_transform(y_preds_scaled)
    
    return y_preds


def preprocess_for_inference(raw_features, artifacts):
    """
    Preprocess raw features for inference using saved scaler.
    
    Args:
        raw_features: Raw feature array or DataFrame
        artifacts: ModelArtifacts instance
        
    Returns:
        scaled_features: Features scaled with the training scaler
    """
    scaler_X = artifacts.scaler_X
    return scaler_X.fit_transform(raw_features)


# Example usage (uncomment to test):
# if __name__ == '__main__':
#     # Load artifacts
#     artifacts = load_model_artifacts('FORD')
#     
#     # Load new data and preprocess
#     new_features = np.random.randn(10, 20)  # Example: 10 samples, 20 features
#     new_features_scaled = preprocess_for_inference(new_features, artifacts)
#     
#     # Make predictions
#     predictions = predict_stock_price(artifacts, new_features_scaled)
#     print(f"Predictions: {predictions}")

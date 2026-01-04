## Production-Ready Model Setup

Your stock prediction models are now production-ready with proper artifact management. Scalers are now saved alongside models for reproducible inference.

---

## 📁 What's New

**New Files Created:**
- `production_utils.py` — Core utilities for saving/loading model artifacts
- `production_inference.py` — Template for making predictions in production
- `README_PRODUCTION.md` — This documentation

**Modified Files:**
- `stock_models/*.py` — All 8 models now save model + scalers as a single "artifact" file

---

## 🔧 How It Works

### Saving Models (Training)

Each model file now saves artifacts automatically at the end of training:

```python
from production_utils import ModelArtifacts

# Create artifacts bundle (model + both scalers)
artifacts = ModelArtifacts(trained_model, scaler_X, scaler_y, ticker='FORD')

# Save to single file
artifacts.save('saved_models/ford_artifacts.joblib')
```

**Output Files (8 total):**
- `saved_models/ford_artifacts.joblib`
- `saved_models/tsla_artifacts.joblib`
- `saved_models/cmcsa_artifacts.joblib`
- `saved_models/fold_artifacts.joblib`
- `saved_models/intc_artifacts.joblib`
- `saved_models/pfe_artifacts.joblib`
- `saved_models/roiv_artifacts.joblib`
- `saved_models/t_artifacts.joblib`

Each artifact file contains:
- Trained PyTorch model
- Feature scaler (MinMaxScaler for inputs)
- Target scaler (MinMaxScaler for outputs)
- Ticker symbol (for tracking)

---

## 🚀 Using Models in Production

### Basic Usage

```python
from production_utils import load_model_artifacts
from production_inference import predict_stock_price, preprocess_for_inference

# Load a model and its scalers
artifacts = load_model_artifacts('FORD', models_dir='saved_models')

# Preprocess raw features using the saved scaler
new_features_scaled = preprocess_for_inference(raw_features, artifacts)

# Make predictions (automatically inverse-transforms back to original scale)
predictions = predict_stock_price(artifacts, new_features_scaled)

print(f"Predicted price: ${predictions[0]:.2f}")
```

### Full Example

```python
import numpy as np
from production_utils import load_model_artifacts
from production_inference import predict_stock_price, preprocess_for_inference

# Load model + scalers (one operation, guarantees consistency)
artifacts = load_model_artifacts('FORD')
print(f"Loaded {artifacts.ticker} model successfully")

# Example: 10 time sequences, 20 features each
new_data = np.random.randn(10, 20)

# Preprocess (scales using the training scaler)
scaled_data = preprocess_for_inference(new_data, artifacts)

# Predict (inverse-transforms automatically)
predictions = predict_stock_price(artifacts, scaled_data)

for i, price in enumerate(predictions):
    print(f"Day {i+1}: ${price[0]:.2f}")
```

---

## 📊 API Reference

### `production_utils.py`

#### `ModelArtifacts` Class

```python
# Create
artifacts = ModelArtifacts(model, scaler_X, scaler_y, ticker='FORD')

# Save
artifacts.save('saved_models/ford_artifacts.joblib')

# Load
artifacts = ModelArtifacts.load('saved_models/ford_artifacts.joblib')

# Access components
model = artifacts.model
scaler_X = artifacts.scaler_X  # For features
scaler_y = artifacts.scaler_y  # For target (stock price)
ticker = artifacts.ticker
```

#### Convenience Functions

```python
# Save artifacts for a ticker
filepath = save_model_artifacts(model, scaler_X, scaler_y, ticker='FORD')

# Load artifacts for a ticker
artifacts = load_model_artifacts('FORD', models_dir='saved_models')
```

### `production_inference.py`

```python
# Preprocess new data using training scaler
scaled_features = preprocess_for_inference(raw_features, artifacts)

# Make predictions with automatic inverse-transform
predictions = predict_stock_price(artifacts, scaled_features, device='cpu')
# predictions are in original stock price scale (not normalized)
```

---

## ✅ Key Benefits

✓ **Consistency**: Scalers are guaranteed to match the model  
✓ **Single File**: One `.joblib` contains everything needed  
✓ **Reproducibility**: No train/test scaler mismatches  
✓ **Easy Deployment**: Load and predict in 2 lines of code  
✓ **All 8 Models**: Ford, Tesla, CMCSA, Fold, INTC, PFE, ROIV, AT&T  

---

## 🔍 What NOT to Do

❌ Don't save/load scalers separately  
❌ Don't mix scalers across different models  
❌ Don't retrain scalers in production  
❌ Don't use unsaved scalers from another train run  

Always use the `ModelArtifacts` class to keep model + scalers in sync.

---

## 📝 Example: Production Inference Service

```python
# app.py - Simple production inference server
from production_utils import load_model_artifacts
from production_inference import predict_stock_price, preprocess_for_inference

# Load all 8 models at startup
models = {}
for ticker in ['F', 'TSLA', 'CMCSA', 'FOLD', 'INTC', 'PFE', 'ROIV', 'T']:
    models[ticker] = load_model_artifacts(ticker)

# Use in request handler
def predict(ticker, features):
    artifacts = models[ticker]
    scaled = preprocess_for_inference(features, artifacts)
    predictions = predict_stock_price(artifacts, scaled)
    return float(predictions[0])
```

---

## 🐛 Troubleshooting

**Q: "ModuleNotFoundError: No module named 'production_utils'"**
- Ensure `production_utils.py` is in the root project directory
- Or add to `PYTHONPATH`

**Q: "No such file: ford_artifacts.joblib"**
- Check file exists in `saved_models/` directory
- Make sure you ran the model training first (creates the file)
- Use `load_model_artifacts('F')` convenience function instead

**Q: Predictions are wrong / scaled strangely**
- Verify you're using `preprocess_for_inference()` for scaling
- Don't manually scale with different scalers
- Check that features have same dimensions as training data (20 features expected)

---

## 🎯 Next Steps

1. **Train all models**: Run each `stock_models/*.py` to generate artifact files
2. **Verify artifacts**: Check `saved_models/` for 8 `.joblib` files
3. **Test inference**: Use `production_inference.py` example
4. **Deploy**: Copy `production_utils.py` and artifact files to production server

---

**Last Updated**: December 28, 2025  
**Status**: Production-Ready ✓

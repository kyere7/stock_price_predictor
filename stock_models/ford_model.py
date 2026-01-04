"""_summary_
A model for predicting Ford stock prices using a hybrid CNN-LSTM-Attention architecture.
"""
import sys
from pathlib import Path

# Add parent directory to path to import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

# imports
from sklearn.ensemble import RandomForestRegressor

# local imports
from scripts.download_data import read_stock_data
from scripts.technical_indicators import add_technical_indicators
from scripts.scripts import train_test_split
from production_utils import ModelArtifacts
from backend.database_service import load_stock_data


# Load data from database (or CSV as fallback)
try:
    f_data = load_stock_data('F')
    if f_data.empty:
        raise Exception("No data from database")
    print("✓ Loaded Ford data from PostgreSQL database")
except Exception as e:
    print(f"Falling back to CSV: {e}")
    f_data = read_stock_data('C:/Users/Gyabaah/Desktop/stock_predictor/stock_data/preprocessed/F.csv')

print(f_data.head())

# add technical indicators
f_data = add_technical_indicators(f_data)
print(f_data.head())

# make the input features dataset and get the target
# Adj Close should be the target, not a feature in the feature_df
f_feature_df = f_data.drop(columns=['Open', 'High', 'Low', 'Volume', 'Adj Close'])
f_data_series = f_data['Adj Close'] # The actual target as a Series

print("F Features DF Head:")
print(f_feature_df.head())
print("\nF Target Series Head:")
print(f_data_series.head())


# Split the data into training and testing sets
f_feature_train, f_feature_test = train_test_split(f_feature_df)
f_target_train, f_target_test = train_test_split(f_data_series) # Split target Series

print("F Feature Train/Test Shapes:", f_feature_train.shape, f_feature_test.shape)
print("F Target Train/Test Shapes:", f_target_train.shape, f_target_test.shape)


# scaling data with separate scalers for features and target
from sklearn.preprocessing import MinMaxScaler

scaler_X = MinMaxScaler() # Scaler for features
scaler_y = MinMaxScaler() # Scaler for target

f_feature_train_scaled = scaler_X.fit_transform(f_feature_train)
f_feature_test_scaled = scaler_X.transform(f_feature_test)

# Target needs to be reshaped to (n_samples, 1) for the scaler
f_target_train_scaled = scaler_y.fit_transform(f_target_train.values.reshape(-1, 1))
f_target_test_scaled = scaler_y.transform(f_target_test.values.reshape(-1, 1))

print("F Scaled Feature Train/Test Shapes:", f_feature_train_scaled.shape, f_feature_test_scaled.shape)
print("F Scaled Target Train/Test Shapes:", f_target_train_scaled.shape, f_target_test_scaled.shape)


# Train a RandomForestRegressor on the (non-sequence) features
# RandomForest expects 2D feature matrix and 1D target
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

# Fit on scaled features and flattened scaled target
rf_model.fit(f_feature_train_scaled, f_target_train_scaled.ravel())

# Predict (scaled)
y_preds_f_scaled = rf_model.predict(f_feature_test_scaled).reshape(-1, 1)

# Inverse transform the predictions and actual values to the original scale
y_preds_f = scaler_y.inverse_transform(y_preds_f_scaled)
y_true_f = scaler_y.inverse_transform(f_target_test_scaled)

print("Predictions generated.")


# Plot the predictions vs actual values
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 7))
plt.plot(y_true_f, label='Actual Price', color='blue')
plt.plot(y_preds_f, label='Predicted Price', color='red', linestyle='--')
plt.title('Ford (F) Stock Price Prediction vs Actual')
plt.xlabel('Time (Days)')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

# Perform evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


mse_f = mean_squared_error(y_true_f, y_preds_f)
rmse_f = np.sqrt(mse_f)
mae_f = mean_absolute_error(y_true_f, y_preds_f)
r2_f = r2_score(y_true_f, y_preds_f)

print(f"F Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse_f:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_f:.4f}")
print(f"Mean Absolute Error (MAE): {mae_f:.4f}")
print(f"R-squared (R2): {r2_f:.4f}")

# Save model and scalers as production artifacts
artifacts = ModelArtifacts(rf_model, scaler_X, scaler_y, ticker='F')
artifacts.save('C:/Users/Gyabaah/Desktop/stock_predictor/saved_models/ford_artifacts.joblib')
print("\n✓ Ford RandomForest model and scalers saved as 'ford_artifacts.joblib'. Ready for production!")
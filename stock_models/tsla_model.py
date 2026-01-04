import sys
from pathlib import Path

# Add parent directory to path to import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

#import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler

# local imports
from scripts.download_data import read_stock_data
from scripts.technical_indicators import add_technical_indicators
from scripts.scripts import train_test_split, create_sequences
from HCLA_model import HybridCNN_LSTM_Attention
from production_utils import ModelArtifacts
from backend.database_service import load_stock_data


try:
    tsla_data = load_stock_data('TSLA')
    if tsla_data.empty:
        raise Exception("No data from database")
    print("✓ Loaded TSLA data from PostgreSQL database")
except Exception as e:
    print(f"Falling back to CSV: {e}")
    tsla_data = read_stock_data('C:/Users/Gyabaah/Desktop/stock_predictor/stock_data/preprocessed/TSLA.csv')
tsla_data = add_technical_indicators(tsla_data)

tsla_feature_df = tsla_data.drop(columns=['Open', 'High', 'Low', 'Volume', 'Adj Close'])
tsla_target_series = tsla_data['Adj Close']

tsla_feature_train, tsla_feature_test = train_test_split(tsla_feature_df)
tsla_target_train, tsla_target_test = train_test_split(tsla_target_series)

scaler_X = MinMaxScaler() # Scaler for features
scaler_y = MinMaxScaler() # Scaler for target

tsla_feature_train_scaled = scaler_X.fit_transform(tsla_feature_train)
tsla_feature_test_scaled = scaler_X.transform(tsla_feature_test)

# Target needs to be reshaped to (n_samples, 1) for the scaler
tsla_target_train_scaled = scaler_y.fit_transform(tsla_target_train.values.reshape(-1, 1))
tsla_target_test_scaled = scaler_y.transform(tsla_target_test.values.reshape(-1, 1))

X_train_tsla, y_train_tsla = create_sequences(tsla_feature_train_scaled, tsla_target_train_scaled)
X_test_tsla, y_test_tsla = create_sequences(tsla_feature_test_scaled, tsla_target_test_scaled)

print("TSLA Feature Train/Test Shapes:", tsla_feature_train.shape, tsla_feature_test.shape)
print("TSLA Target Train/Test Shapes:", tsla_target_train.shape, tsla_target_test.shape)
print("TSLA Scaled Feature Train/Test Shapes:", tsla_feature_train_scaled.shape, tsla_feature_test_scaled.shape)
print("TSLA Scaled Target Train/Test Shapes:", tsla_target_train_scaled.shape, tsla_target_test_scaled.shape)
print("TSLA Sequence X_train_tsla shape:", X_train_tsla.shape)
print("TSLA Sequence y_train_tsla shape:", y_train_tsla.shape)
print("TSLA Sequence X_test_tsla shape:", X_test_tsla.shape)
print("TSLA Sequence y_test_tsla shape:", y_test_tsla.shape)

# Initialize the Hybrid CNN-LSTM Attention model for TSLA
tsla_model = HybridCNN_LSTM_Attention(input_features=X_train_tsla.shape[2], cnn_filters=128, lstm_hidden_size=128)

#Define optimizer
optimizer = optim.Adam(tsla_model.parameters(), lr=0.001)

#Define Loss Function (Mean Squared Error for regression)
criterion = nn.MSELoss()

print("Hybrid CNN-LSTM Attention Model for TSLA:")
print(tsla_model)

print("\nOptimizer:", optimizer)
print("\nLoss Function:", criterion)


# Convert numpy arrays to PyTorch tensors
X_train_tsla_tensor = torch.from_numpy(X_train_tsla).float()
y_train_tsla_tensor = torch.from_numpy(y_train_tsla).float() # Assuming y_train_tsla is already (N, 1) or (N,) and MSELoss handles it

# Create a TensorDataset and DataLoader
train_dataset_tsla = data.TensorDataset(X_train_tsla_tensor, y_train_tsla_tensor)
train_loader_tsla = data.DataLoader(dataset=train_dataset_tsla, batch_size=32, shuffle=False)

# Define the number of epochs
num_epochs = 50 # Using 50 epochs as before

# Move model to device if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tsla_model.to(device)

print(f"Training TSLA model on {device}")

# Training loop
for epoch in range(num_epochs):
    tsla_model.train() # Set the model to training mode
    for batch_idx, (X_batch_tsla, y_batch_tsla) in enumerate(train_loader_tsla):
        X_batch_tsla, y_batch_tsla = X_batch_tsla.to(device), y_batch_tsla.to(device)

        # Forward pass
        outputs_tsla = tsla_model(X_batch_tsla)
        loss = criterion(outputs_tsla, y_batch_tsla)

        # Backward and optimize
        optimizer.zero_grad() # Clear gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update weights

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("TSLA model training complete!")

# predicting and plotting results
import matplotlib.pyplot as plt

# Put the model in evaluation mode
tsla_model.eval()

# Move X_test to the same device as the model
X_test_tsla_tensor = torch.from_numpy(X_test_tsla).float().to(device)

# Make predictions
with torch.no_grad(): # Disable gradient calculations for inference
    y_preds_tsla_scaled = tsla_model(X_test_tsla_tensor).cpu().numpy()

# Inverse transform the predictions and actual values to the original scale
y_preds_tsla = scaler_y.inverse_transform(y_preds_tsla_scaled) # Use scaler_y
y_true_tsla = scaler_y.inverse_transform(y_test_tsla) # y_test is already (N,1) if from create_sequences with 2nd arg

# Plot the results
plt.figure(figsize=(15, 7))
plt.plot(y_true_tsla, label='Actual Price', color='blue')
plt.plot(y_preds_tsla, label='Predicted Price', color='red', linestyle='--')
plt.title('TSLA Stock Price Prediction vs Actual')
plt.xlabel('Time (Days)')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

print("TSLA predictions generated and plot displayed.")

# evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse_tsla = mean_squared_error(y_true_tsla, y_preds_tsla)
rmse_tsla = np.sqrt(mse_tsla)
mae_tsla = mean_absolute_error(y_true_tsla, y_preds_tsla)
r2_tsla = r2_score(y_true_tsla, y_preds_tsla)

print(f"TSLA Hybrid CNN-LSTM Attention Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse_tsla:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_tsla:.4f}")
print(f"Mean Absolute Error (MAE): {mae_tsla:.4f}")
print(f"R-squared (R2): {r2_tsla:.4f}")

# Save model and scalers as production artifacts
artifacts = ModelArtifacts(tsla_model, scaler_X, scaler_y, ticker='TSLA')
artifacts.save('C:/Users/Gyabaah/Desktop/stock_predictor/saved_models/tsla_artifacts.joblib')
print("\n✓ TSLA model and scalers saved as 'tsla_artifacts.joblib'. Ready for production!")
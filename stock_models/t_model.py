import sys
from pathlib import Path

# Add parent directory to path to import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

# imports
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# local imports
from scripts.download_data import read_stock_data
from scripts.technical_indicators import add_technical_indicators
from scripts.scripts import train_test_split, create_sequences
from HCLA_model import HybridCNN_LSTM_Attention
from production_utils import ModelArtifacts
from backend.database_service import load_stock_data


try:
    t_data = load_stock_data('T')
    if t_data.empty:
        raise Exception("No data from database")
    print("✓ Loaded AT&T data from PostgreSQL database")
except Exception as e:
    print(f"Falling back to CSV: {e}")
    t_data = read_stock_data('C:/Users/Gyabaah/Desktop/stock_predictor/stock_data/preprocessed/T.csv')
t_data = add_technical_indicators(t_data)

t_feature_df = t_data.drop(columns=['Open', 'High', 'Low', 'Volume', 'Adj Close'])
t_target_series = t_data['Adj Close']

t_feature_train, t_feature_test = train_test_split(t_feature_df)
t_target_train, t_target_test = train_test_split(t_target_series)

scaler_X = MinMaxScaler() # Scaler for features
scaler_y = MinMaxScaler() # Scaler for target

t_feature_train_scaled = scaler_X.fit_transform(t_feature_train)
t_feature_test_scaled = scaler_X.transform(t_feature_test)

# Target needs to be reshaped to (n_samples, 1) for the scaler
t_target_train_scaled = scaler_y.fit_transform(t_target_train.values.reshape(-1, 1))
t_target_test_scaled = scaler_y.transform(t_target_test.values.reshape(-1, 1))

X_train_t, y_train_t = create_sequences(t_feature_train_scaled, t_target_train_scaled)
X_test_t, y_test_t = create_sequences(t_feature_test_scaled, t_target_test_scaled)

print("T Feature Train/Test Shapes:", t_feature_train.shape, t_feature_test.shape)
print("T Target Train/Test Shapes:", t_target_train.shape, t_target_test.shape)
print("T Scaled Feature Train/Test Shapes:", t_feature_train_scaled.shape, t_feature_test_scaled.shape)
print("T Scaled Target Train/Test Shapes:", t_target_train_scaled.shape, t_target_test_scaled.shape)
print("T Sequence X_train_t shape:", X_train_t.shape)
print("T Sequence y_train_t shape:", y_train_t.shape)
print("T Sequence X_test_t shape:", X_test_t.shape)
print("T Sequence y_test_t shape:", y_test_t.shape)

# Initialize the Hybrid CNN-LSTM Attention model for AT&T
t_model = HybridCNN_LSTM_Attention(input_features=X_train_t.shape[2], cnn_filters=128, lstm_hidden_size=128)

#Define optimizer
optimizer = optim.Adam(t_model.parameters(), lr=0.001)

#Define Loss Function (Mean Squared Error for regression)
criterion = nn.MSELoss()

print("Hybrid CNN-LSTM Attention Model for AT&T:")
print(t_model)

print("\nOptimizer:", optimizer)
print("\nLoss Function:", criterion)

# Training the AT&T model
import torch.utils.data as data

# Convert numpy arrays to PyTorch tensors
X_train_t_tensor = torch.from_numpy(X_train_t).float()
y_train_t_tensor = torch.from_numpy(y_train_t).float() # Assuming y_train_t is already (N, 1) or (N,) and MSELoss handles it

# Create a TensorDataset and DataLoader
train_dataset_t = data.TensorDataset(X_train_t_tensor, y_train_t_tensor)
train_loader_t = data.DataLoader(dataset=train_dataset_t, batch_size=32, shuffle=False)

# Define the number of epochs
num_epochs = 50 # Using 50 epochs as before

# Move model to device if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t_model.to(device)

print(f"Training AT&T model on {device}")

# Training loop
for epoch in range(num_epochs):
    t_model.train() # Set the model to training mode
    for batch_idx, (X_batch_t, y_batch_t) in enumerate(train_loader_t):
        X_batch_t, y_batch_t = X_batch_t.to(device), y_batch_t.to(device)

        # Forward pass
        outputs_t = t_model(X_batch_t)
        loss = criterion(outputs_t, y_batch_t)

        # Backward and optimize
        optimizer.zero_grad() # Clear gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update weights

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("AT&T model training complete!")

import matplotlib.pyplot as plt

# Put the model in evaluation mode
t_model.eval()

# Move X_test to the same device as the model
X_test_t_tensor = torch.from_numpy(X_test_t).float().to(device)

# Make predictions
with torch.no_grad(): # Disable gradient calculations for inference
    y_preds_t_scaled = t_model(X_test_t_tensor).cpu().numpy()

# Inverse transform the predictions and actual values to the original scale
y_preds_t = scaler_y.inverse_transform(y_preds_t_scaled) # Use scaler_y
y_true_t = scaler_y.inverse_transform(y_test_t) # y_test is already (N,1) if from create_sequences with 2nd arg

# Plot the results
plt.figure(figsize=(15, 7))
plt.plot(y_true_t, label='Actual Price', color='blue')
plt.plot(y_preds_t, label='Predicted Price', color='red', linestyle='--')
plt.title('AT&T (T) Stock Price Prediction vs Actual')
plt.xlabel('Time (Days)')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

print("AT&T predictions generated and plot displayed.")

# evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse_t = mean_squared_error(y_true_t, y_preds_t)
rmse_t = np.sqrt(mse_t)
mae_t = mean_absolute_error(y_true_t, y_preds_t)
r2_t = r2_score(y_true_t, y_preds_t)

print(f"AT&T Hybrid CNN-LSTM Attention Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse_t:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_t:.4f}")
print(f"Mean Absolute Error (MAE): {mae_t:.4f}")
print(f"R-squared (R2): {r2_t:.4f}")

# Save model and scalers as production artifacts
artifacts = ModelArtifacts(t_model, scaler_X, scaler_y, ticker='T')
artifacts.save('C:/Users/Gyabaah/Desktop/stock_predictor/saved_models/t_artifacts.joblib')
print("\n✓ AT&T model and scalers saved as 't_artifacts.joblib'. Ready for production!")

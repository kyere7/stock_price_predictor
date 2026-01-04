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
    intc_data = load_stock_data('INTC')
    if intc_data.empty:
        raise Exception("No data from database")
    print("✓ Loaded INTC data from PostgreSQL database")
except Exception as e:
    print(f"Falling back to CSV: {e}")
    intc_data = read_stock_data('C:/Users/Gyabaah/Desktop/stock_predictor/stock_data/preprocessed/INTC.csv')
intc_data = add_technical_indicators(intc_data)

intc_feature_df = intc_data.drop(columns=['Open', 'High', 'Low', 'Volume', 'Adj Close'])
intc_target_series = intc_data['Adj Close']

intc_feature_train, intc_feature_test = train_test_split(intc_feature_df)
intc_target_train, intc_target_test = train_test_split(intc_target_series)

scaler_X = MinMaxScaler() # Scaler for features
scaler_y = MinMaxScaler() # Scaler for target

intc_feature_train_scaled = scaler_X.fit_transform(intc_feature_train)
intc_feature_test_scaled = scaler_X.transform(intc_feature_test)

# Target needs to be reshaped to (n_samples, 1) for the scaler
intc_target_train_scaled = scaler_y.fit_transform(intc_target_train.values.reshape(-1, 1))
intc_target_test_scaled = scaler_y.transform(intc_target_test.values.reshape(-1, 1))

X_train_intc, y_train_intc = create_sequences(intc_feature_train_scaled, intc_target_train_scaled)
X_test_intc, y_test_intc = create_sequences(intc_feature_test_scaled, intc_target_test_scaled)

print("INTC Feature Train/Test Shapes:", intc_feature_train.shape, intc_feature_test.shape)
print("INTC Target Train/Test Shapes:", intc_target_train.shape, intc_target_test.shape)
print("INTC Scaled Feature Train/Test Shapes:", intc_feature_train_scaled.shape, intc_feature_test_scaled.shape)
print("INTC Scaled Target Train/Test Shapes:", intc_target_train_scaled.shape, intc_target_test_scaled.shape)
print("INTC Sequence X_train_intc shape:", X_train_intc.shape)
print("INTC Sequence y_train_intc shape:", y_train_intc.shape)
print("INTC Sequence X_test_intc shape:", X_test_intc.shape)
print("INTC Sequence y_test_intc shape:", y_test_intc.shape)

intc_model = HybridCNN_LSTM_Attention(input_features=X_train_intc.shape[2], cnn_filters=128, lstm_hidden_size=128)

#Define optimizer
optimizer = optim.Adam(intc_model.parameters(), lr=0.001)

#Define Loss Function (Mean Squared Error for regression)
criterion = nn.MSELoss()

print("Hybrid CNN-LSTM Attention Model for INTC:")
print(intc_model)

print("\nOptimizer:", optimizer)
print("\nLoss Function:", criterion)

import torch.utils.data as data

# Convert numpy arrays to PyTorch tensors
X_train_intc_tensor = torch.from_numpy(X_train_intc).float()
y_train_intc_tensor = torch.from_numpy(y_train_intc).float() # Assuming y_train_intc is already (N, 1) or (N,) and MSELoss handles it

# Create a TensorDataset and DataLoader
train_dataset_intc = data.TensorDataset(X_train_intc_tensor, y_train_intc_tensor)
train_loader_intc = data.DataLoader(dataset=train_dataset_intc, batch_size=32, shuffle=False)

# Define the number of epochs
num_epochs = 50 # Using 50 epochs as before

# Move model to device if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
intc_model.to(device)

print(f"Training INTC model on {device}")

# Training loop
for epoch in range(num_epochs):
    intc_model.train() # Set the model to training mode
    for batch_idx, (X_batch_intc, y_batch_intc) in enumerate(train_loader_intc):
        X_batch_intc, y_batch_intc = X_batch_intc.to(device), y_batch_intc.to(device)

        # Forward pass
        outputs_intc = intc_model(X_batch_intc)
        loss = criterion(outputs_intc, y_batch_intc)

        # Backward and optimize
        optimizer.zero_grad() # Clear gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update weights

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("INTC model training complete!")

# Evaluation and Predictions
import matplotlib.pyplot as plt

# Put the model in evaluation mode
intc_model.eval()

# Move X_test to the same device as the model
X_test_intc_tensor = torch.from_numpy(X_test_intc).float().to(device)

# Make predictions
with torch.no_grad(): # Disable gradient calculations for inference
    y_preds_intc_scaled = intc_model(X_test_intc_tensor).cpu().numpy()

# Inverse transform the predictions and actual values to the original scale
y_preds_intc = scaler_y.inverse_transform(y_preds_intc_scaled) # Use scaler_y
y_true_intc = scaler_y.inverse_transform(y_test_intc) # y_test is already (N,1) if from create_sequences with 2nd arg

# Plot the results
plt.figure(figsize=(15, 7))
plt.plot(y_true_intc, label='Actual Price', color='blue')
plt.plot(y_preds_intc, label='Predicted Price', color='red', linestyle='--')
plt.title('INTC Stock Price Prediction vs Actual')
plt.xlabel('Time (Days)')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

print("INTC predictions generated and plot displayed.")

# evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse_intc = mean_squared_error(y_true_intc, y_preds_intc)
rmse_intc = np.sqrt(mse_intc)
mae_intc = mean_absolute_error(y_true_intc, y_preds_intc)
r2_intc = r2_score(y_true_intc, y_preds_intc)

print(f"INTC Hybrid CNN-LSTM Attention Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse_intc:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_intc:.4f}")
print(f"Mean Absolute Error (MAE): {mae_intc:.4f}")
print(f"R-squared (R2): {r2_intc:.4f}")

# Save model and scalers as production artifacts
artifacts = ModelArtifacts(intc_model, scaler_X, scaler_y, ticker='INTC')
artifacts.save('C:/Users/Gyabaah/Desktop/stock_predictor/saved_models/intc_artifacts.joblib')
print("\n✓ INTC model and scalers saved as 'intc_artifacts.joblib'. Ready for production!")

import sys
from pathlib import Path

# Add parent directory to path to import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# local imports
from scripts.download_data import read_stock_data
from scripts.technical_indicators import add_technical_indicators
from scripts.scripts import train_test_split, create_sequences
from HCLA_model import HybridCNN_LSTM_Attention
from production_utils import ModelArtifacts
from backend.database_service import load_stock_data


try:
    pfe_data = load_stock_data('PFE')
    if pfe_data.empty:
        raise Exception("No data from database")
    print("✓ Loaded PFE data from PostgreSQL database")
except Exception as e:
    print(f"Falling back to CSV: {e}")
    pfe_data = read_stock_data('C:/Users/Gyabaah/Desktop/stock_predictor/stock_data/preprocessed/PFE.csv')
pfe_data = add_technical_indicators(pfe_data)

pfe_feature_df = pfe_data.drop(columns=['Open', 'High', 'Low', 'Volume', 'Adj Close'])
pfe_target_series = pfe_data['Adj Close']

pfe_feature_train, pfe_feature_test = train_test_split(pfe_feature_df)
pfe_target_train, pfe_target_test = train_test_split(pfe_target_series)

scaler_X = MinMaxScaler() # Scaler for features
scaler_y = MinMaxScaler() # Scaler for target

pfe_feature_train_scaled = scaler_X.fit_transform(pfe_feature_train)
pfe_feature_test_scaled = scaler_X.transform(pfe_feature_test)

# Target needs to be reshaped to (n_samples, 1) for the scaler
pfe_target_train_scaled = scaler_y.fit_transform(pfe_target_train.values.reshape(-1, 1))
pfe_target_test_scaled = scaler_y.transform(pfe_target_test.values.reshape(-1, 1))

X_train_pfe, y_train_pfe = create_sequences(pfe_feature_train_scaled, pfe_target_train_scaled)
X_test_pfe, y_test_pfe = create_sequences(pfe_feature_test_scaled, pfe_target_test_scaled)

print("PFE Feature Train/Test Shapes:", pfe_feature_train.shape, pfe_feature_test.shape)
print("PFE Target Train/Test Shapes:", pfe_target_train.shape, pfe_target_test.shape)
print("PFE Scaled Feature Train/Test Shapes:", pfe_feature_train_scaled.shape, pfe_feature_test_scaled.shape)
print("PFE Scaled Target Train/Test Shapes:", pfe_target_train_scaled.shape, pfe_target_test_scaled.shape)
print("PFE Sequence X_train_pfe shape:", X_train_pfe.shape)
print("PFE Sequence y_train_pfe shape:", y_train_pfe.shape)
print("PFE Sequence X_test_pfe shape:", X_test_pfe.shape)
print("PFE Sequence y_test_pfe shape:", y_test_pfe.shape)


pfe_model = HybridCNN_LSTM_Attention(input_features=X_train_pfe.shape[2], cnn_filters=128, lstm_hidden_size=128)

#Define optimizer
optimizer = optim.Adam(pfe_model.parameters(), lr=0.001)

#Define Loss Function (Mean Squared Error for regression)
criterion = nn.MSELoss()

print("Hybrid CNN-LSTM Attention Model for PFE:")
print(pfe_model)

print("\nOptimizer:", optimizer)
print("\nLoss Function:", criterion)


import torch.utils.data as data

# Convert numpy arrays to PyTorch tensors
X_train_pfe_tensor = torch.from_numpy(X_train_pfe).float()
y_train_pfe_tensor = torch.from_numpy(y_train_pfe).float() # Assuming y_train_pfe is already (N, 1) or (N,) and MSELoss handles it

# Create a TensorDataset and DataLoader
train_dataset_pfe = data.TensorDataset(X_train_pfe_tensor, y_train_pfe_tensor)
train_loader_pfe = data.DataLoader(dataset=train_dataset_pfe, batch_size=32, shuffle=False)

# Define the number of epochs
num_epochs = 50 # Using 50 epochs as before

# Move model to device if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pfe_model.to(device)

print(f"Training PFE model on {device}")

# Training loop
for epoch in range(num_epochs):
    pfe_model.train() # Set the model to training mode
    for batch_idx, (X_batch_pfe, y_batch_pfe) in enumerate(train_loader_pfe):
        X_batch_pfe, y_batch_pfe = X_batch_pfe.to(device), y_batch_pfe.to(device)

        # Forward pass
        outputs_pfe = pfe_model(X_batch_pfe)
        loss = criterion(outputs_pfe, y_batch_pfe)

        # Backward and optimize
        optimizer.zero_grad() # Clear gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update weights

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("PFE model training complete!")

# Evaluation
pfe_model.eval()

X_test_pfe_tensor = torch.from_numpy(X_test_pfe).float().to(device)

with torch.no_grad():
    y_preds_pfe_scaled = pfe_model(X_test_pfe_tensor).cpu().numpy()

y_preds_pfe = scaler_y.inverse_transform(y_preds_pfe_scaled)
y_true_pfe = scaler_y.inverse_transform(y_test_pfe)

plt.figure(figsize=(15, 7))
plt.plot(y_true_pfe, label='Actual Price', color='blue')
plt.plot(y_preds_pfe, label='Predicted Price', color='red', linestyle='--')
plt.title('PFE Stock Price Prediction vs Actual')
plt.xlabel('Time (Days)')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

print("PFE predictions generated and plot displayed.")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse_pfe = mean_squared_error(y_true_pfe, y_preds_pfe)
rmse_pfe = np.sqrt(mse_pfe)
mae_pfe = mean_absolute_error(y_true_pfe, y_preds_pfe)
r2_pfe = r2_score(y_true_pfe, y_preds_pfe)

print(f"PFE Hybrid CNN-LSTM Attention Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse_pfe:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_pfe:.4f}")
print(f"Mean Absolute Error (MAE): {mae_pfe:.4f}")
print(f"R-squared (R2): {r2_pfe:.4f}")

# Save model and scalers as production artifacts
artifacts = ModelArtifacts(pfe_model, scaler_X, scaler_y, ticker='PFE')
artifacts.save('C:/Users/Gyabaah/Desktop/stock_predictor/saved_models/pfe_artifacts.joblib')
print("\n✓ PFE model and scalers saved as 'pfe_artifacts.joblib'. Ready for production!")

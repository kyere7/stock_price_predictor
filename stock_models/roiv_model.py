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

roiv_data = load_stock_data('ROIV')
if roiv_data.empty:
    print("Falling back to CSV for ROIV")
    roiv_data = read_stock_data('C:/Users/Gyabaah/Desktop/stock_predictor/stock_data/preprocessed/ROIV.csv')
else:
    print("✓ Loaded ROIV data from PostgreSQL database")
roiv_data = add_technical_indicators(roiv_data)

roiv_feature_df = roiv_data.drop(columns=['Open', 'High', 'Low', 'Volume', 'Adj Close'])
roiv_target_series = roiv_data['Adj Close']

roiv_feature_train, roiv_feature_test = train_test_split(roiv_feature_df)
roiv_target_train, roiv_target_test = train_test_split(roiv_target_series)

scaler_X = MinMaxScaler() # Scaler for features
scaler_y = MinMaxScaler() # Scaler for target

roiv_feature_train_scaled = scaler_X.fit_transform(roiv_feature_train)
roiv_feature_test_scaled = scaler_X.transform(roiv_feature_test)

# Target needs to be reshaped to (n_samples, 1) for the scaler
roiv_target_train_scaled = scaler_y.fit_transform(roiv_target_train.values.reshape(-1, 1))
roiv_target_test_scaled = scaler_y.transform(roiv_target_test.values.reshape(-1, 1))

X_train_roiv, y_train_roiv = create_sequences(roiv_feature_train_scaled, roiv_target_train_scaled)
X_test_roiv, y_test_roiv = create_sequences(roiv_feature_test_scaled, roiv_target_test_scaled)

print("ROIV Feature Train/Test Shapes:", roiv_feature_train.shape, roiv_feature_test.shape)
print("ROIV Target Train/Test Shapes:", roiv_target_train.shape, roiv_target_test.shape)
print("ROIV Scaled Feature Train/Test Shapes:", roiv_feature_train_scaled.shape, roiv_feature_test_scaled.shape)
print("ROIV Scaled Target Train/Test Shapes:", roiv_target_train_scaled.shape, roiv_target_test_scaled.shape)
print("ROIV Sequence X_train_roiv shape:", X_train_roiv.shape)
print("ROIV Sequence y_train_roiv shape:", y_train_roiv.shape)
print("ROIV Sequence X_test_roiv shape:", X_test_roiv.shape)
print("ROIV Sequence y_test_roiv shape:", y_test_roiv.shape)

# define the Hybrid CNN-LSTM Attention model for ROIV
roiv_model = HybridCNN_LSTM_Attention(input_features=X_train_roiv.shape[2], cnn_filters=128, lstm_hidden_size=128)

#Define optimizer
optimizer = optim.Adam(roiv_model.parameters(), lr=0.001)

#Define Loss Function (Mean Squared Error for regression)
criterion = nn.MSELoss()

print("Hybrid CNN-LSTM Attention Model for ROIV:")
print(roiv_model)

print("\nOptimizer:", optimizer)
print("\nLoss Function:", criterion)


# Training the ROIV model
# Convert numpy arrays to PyTorch tensors
import torch.utils.data as data

X_train_roiv_tensor = torch.from_numpy(X_train_roiv).float()
y_train_roiv_tensor = torch.from_numpy(y_train_roiv).float() # Assuming y_train_roiv is already (N, 1) or (N,) and MSELoss handles it

# Create a TensorDataset and DataLoader
train_dataset_roiv = data.TensorDataset(X_train_roiv_tensor, y_train_roiv_tensor)
train_loader_roiv = data.DataLoader(dataset=train_dataset_roiv, batch_size=32, shuffle=False)

# Define the number of epochs
num_epochs = 50 # Using 50 epochs as before

# Move model to device if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
roiv_model.to(device)

print(f"Training ROIV model on {device}")

# Training loop
for epoch in range(num_epochs):
    roiv_model.train() # Set the model to training mode
    for batch_idx, (X_batch_roiv, y_batch_roiv) in enumerate(train_loader_roiv):
        X_batch_roiv, y_batch_roiv = X_batch_roiv.to(device), y_batch_roiv.to(device)

        # Forward pass
        outputs_roiv = roiv_model(X_batch_roiv)
        loss = criterion(outputs_roiv, y_batch_roiv)

        # Backward and optimize
        optimizer.zero_grad() # Clear gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update weights

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("ROIV model training complete!")

# evaluation and predictions
import matplotlib.pyplot as plt

# Put the model in evaluation mode
roiv_model.eval()

# Move X_test to the same device as the model
X_test_roiv_tensor = torch.from_numpy(X_test_roiv).float().to(device)

# Make predictions
with torch.no_grad(): # Disable gradient calculations for inference
    y_preds_roiv_scaled = roiv_model(X_test_roiv_tensor).cpu().numpy()

# Inverse transform the predictions and actual values to the original scale
y_preds_roiv = scaler_y.inverse_transform(y_preds_roiv_scaled) # Use scaler_y
y_true_roiv = scaler_y.inverse_transform(y_test_roiv) # y_test is already (N,1) if from create_sequences with 2nd arg

# Plot the results
plt.figure(figsize=(15, 7))
plt.plot(y_true_roiv, label='Actual Price', color='blue')
plt.plot(y_preds_roiv, label='Predicted Price', color='red', linestyle='--')
plt.title('ROIV Stock Price Prediction vs Actual')
plt.xlabel('Time (Days)')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

print("ROIV predictions generated and plot displayed.")

# evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse_roiv = mean_squared_error(y_true_roiv, y_preds_roiv)
rmse_roiv = np.sqrt(mse_roiv)
mae_roiv = mean_absolute_error(y_true_roiv, y_preds_roiv)
r2_roiv = r2_score(y_true_roiv, y_preds_roiv)

print(f"ROIV Hybrid CNN-LSTM Attention Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse_roiv:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_roiv:.4f}")
print(f"Mean Absolute Error (MAE): {mae_roiv:.4f}")
print(f"R-squared (R2): {r2_roiv:.4f}")

# Save model and scalers as production artifacts
artifacts = ModelArtifacts(roiv_model, scaler_X, scaler_y, ticker='ROIV')
artifacts.save('C:/Users/Gyabaah/Desktop/stock_predictor/saved_models/roiv_artifacts.joblib')
print("\n✓ ROIV model and scalers saved as 'roiv_artifacts.joblib'. Ready for production!")

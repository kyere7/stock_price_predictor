import sys
from pathlib import Path

# Add parent directory to path to import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

#import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# local imports
from scripts.download_data import read_stock_data
from scripts.technical_indicators import add_technical_indicators
from scripts.scripts import train_test_split, create_sequences
from HCLA_model import HybridCNN_LSTM_Attention
from production_utils import ModelArtifacts
from backend.database_service import load_stock_data


try:
    fold_data = load_stock_data('FOLD')
    if fold_data.empty:
        raise Exception("No data from database")
    print("✓ Loaded FOLD data from PostgreSQL database")
except Exception as e:
    print(f"Falling back to CSV: {e}")
    fold_data = read_stock_data('C:/Users/Gyabaah/Desktop/stock_predictor/stock_data/preprocessed/FOLD.csv')
fold_data = add_technical_indicators(fold_data)

fold_feature_df = fold_data.drop(columns=['Open', 'High', 'Low', 'Volume', 'Adj Close'])
fold_target_series = fold_data['Adj Close']

fold_feature_train, fold_feature_test = train_test_split(fold_feature_df)
fold_target_train, fold_target_test = train_test_split(fold_target_series)

scaler_X = MinMaxScaler() # Scaler for features
scaler_y = MinMaxScaler() # Scaler for target

fold_feature_train_scaled = scaler_X.fit_transform(fold_feature_train)
fold_feature_test_scaled = scaler_X.transform(fold_feature_test)

# Target needs to be reshaped to (n_samples, 1) for the scaler
fold_target_train_scaled = scaler_y.fit_transform(fold_target_train.values.reshape(-1, 1))
fold_target_test_scaled = scaler_y.transform(fold_target_test.values.reshape(-1, 1))

X_train_fold, y_train_fold = create_sequences(fold_feature_train_scaled, fold_target_train_scaled)
X_test_fold, y_test_fold = create_sequences(fold_feature_test_scaled, fold_target_test_scaled)

print("FOLD Feature Train/Test Shapes:", fold_feature_train.shape, fold_feature_test.shape)
print("FOLD Target Train/Test Shapes:", fold_target_train.shape, fold_target_test.shape)
print("FOLD Scaled Feature Train/Test Shapes:", fold_feature_train_scaled.shape, fold_feature_test_scaled.shape)
print("FOLD Scaled Target Train/Test Shapes:", fold_target_train_scaled.shape, fold_target_test_scaled.shape)
print("FOLD Sequence X_train_fold shape:", X_train_fold.shape)
print("FOLD Sequence y_train_fold shape:", y_train_fold.shape)
print("FOLD Sequence X_test_fold shape:", X_test_fold.shape)
print("FOLD Sequence y_test_fold shape:", y_test_fold.shape)

# define the Hybrid CNN-LSTM Attention model for FOLD
fold_model = HybridCNN_LSTM_Attention(input_features=X_train_fold.shape[2], cnn_filters=128, lstm_hidden_size=128)

#Define optimizer
optimizer = optim.Adam(fold_model.parameters(), lr=0.001)

#Define Loss Function (Mean Squared Error for regression)
criterion = nn.MSELoss()

print("Hybrid CNN-LSTM Attention Model for FOLD:")
print(fold_model)

print("\nOptimizer:", optimizer)
print("\nLoss Function:", criterion)

# Training the FOLD model
import torch.utils.data as data

# Convert numpy arrays to PyTorch tensors
X_train_fold_tensor = torch.from_numpy(X_train_fold).float()
y_train_fold_tensor = torch.from_numpy(y_train_fold).float() # Assuming y_train_fold is already (N, 1) or (N,) and MSELoss handles it

# Create a TensorDataset and DataLoader
train_dataset_fold = data.TensorDataset(X_train_fold_tensor, y_train_fold_tensor)
train_loader_fold = data.DataLoader(dataset=train_dataset_fold, batch_size=32, shuffle=False)

# Define the number of epochs
num_epochs = 50 # Using 50 epochs as before

# Move model to device if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fold_model.to(device)

print(f"Training FOLD model on {device}")

# Training loop
for epoch in range(num_epochs):
    fold_model.train() # Set the model to training mode
    for batch_idx, (X_batch_fold, y_batch_fold) in enumerate(train_loader_fold):
        X_batch_fold, y_batch_fold = X_batch_fold.to(device), y_batch_fold.to(device)

        # Forward pass
        outputs_fold = fold_model(X_batch_fold)
        loss = criterion(outputs_fold, y_batch_fold)

        # Backward and optimize
        optimizer.zero_grad() # Clear gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update weights

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("FOLD model training complete!")


# Evaluation
import matplotlib.pyplot as plt

# Put the model in evaluation mode
fold_model.eval()

# Move X_test to the same device as the model
X_test_fold_tensor = torch.from_numpy(X_test_fold).float().to(device)

# Make predictions
with torch.no_grad(): # Disable gradient calculations for inference
    y_preds_fold_scaled = fold_model(X_test_fold_tensor).cpu().numpy()

# Inverse transform the predictions and actual values to the original scale
y_preds_fold = scaler_y.inverse_transform(y_preds_fold_scaled) # Use scaler_y
y_true_fold = scaler_y.inverse_transform(y_test_fold) # y_test is already (N,1) if from create_sequences with 2nd arg

# Plot the results
plt.figure(figsize=(15, 7))
plt.plot(y_true_fold, label='Actual Price', color='blue')
plt.plot(y_preds_fold, label='Predicted Price', color='red', linestyle='--')
plt.title('FOLD Stock Price Prediction vs Actual')
plt.xlabel('Time (Days)')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

print("FOLD predictions generated and plot displayed.")

# evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse_fold = mean_squared_error(y_true_fold, y_preds_fold)
rmse_fold = np.sqrt(mse_fold)
mae_fold = mean_absolute_error(y_true_fold, y_preds_fold)
r2_fold = r2_score(y_true_fold, y_preds_fold)

print(f"FOLD Hybrid CNN-LSTM Attention Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse_fold:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_fold:.4f}")
print(f"Mean Absolute Error (MAE): {mae_fold:.4f}")
print(f"R-squared (R2): {r2_fold:.4f}")

# Save model and scalers as production artifacts
artifacts = ModelArtifacts(fold_model, scaler_X, scaler_y, ticker='FOLD')
artifacts.save('C:/Users/Gyabaah/Desktop/stock_predictor/saved_models/fold_artifacts.joblib')
print("\n✓ FOLD model and scalers saved as 'fold_artifacts.joblib'. Ready for production!")

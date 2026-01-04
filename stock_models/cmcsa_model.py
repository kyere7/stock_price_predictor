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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# local imports
from scripts.download_data import read_stock_data
from scripts.technical_indicators import add_technical_indicators
from scripts.scripts import train_test_split, create_sequences
from HCLA_model import HybridCNN_LSTM_Attention
from production_utils import ModelArtifacts
from backend.database_service import load_stock_data


def main():
    try:
        cmcsa_data = load_stock_data('CMCSA')
        if cmcsa_data.empty:
            raise Exception("No data from database")
        print("✓ Loaded CMCSA data from PostgreSQL database")
    except Exception as e:
        print(f"Falling back to CSV: {e}")
        cmcsa_data = read_stock_data('C:/Users/Gyabaah/Desktop/stock_predictor/stock_data/preprocessed/CMCSA.csv')
    cmcsa_data = add_technical_indicators(cmcsa_data)

    cmcsa_feature_df = cmcsa_data.drop(columns=['Open', 'High', 'Low', 'Volume', 'Adj Close'])
    cmcsa_target_series = cmcsa_data['Adj Close']

    # Split the data into training and testing sets
    cmcsa_feature_train, cmcsa_feature_test = train_test_split(cmcsa_feature_df)
    cmcsa_target_train, cmcsa_target_test = train_test_split(cmcsa_target_series)

    scaler_X = MinMaxScaler() # Scaler for features
    scaler_y = MinMaxScaler() # Scaler for target

    cmcsa_feature_train_scaled = scaler_X.fit_transform(cmcsa_feature_train)
    cmcsa_feature_test_scaled = scaler_X.transform(cmcsa_feature_test)

    # Target needs to be reshaped to (n_samples, 1) for the scaler
    cmcsa_target_train_scaled = scaler_y.fit_transform(cmcsa_target_train.values.reshape(-1, 1))
    cmcsa_target_test_scaled = scaler_y.transform(cmcsa_target_test.values.reshape(-1, 1))

    # sequence construction
    X_train_cmcsa, y_train_cmcsa = create_sequences(cmcsa_feature_train_scaled, cmcsa_target_train_scaled)
    X_test_cmcsa, y_test_cmcsa = create_sequences(cmcsa_feature_test_scaled, cmcsa_target_test_scaled)

    # Print the heads of the dataframes for verification
    print("CMCSA Features DF Head:")
    print(cmcsa_feature_df.head())
    print("\nCMCSA Target Series Head:")
    print(cmcsa_target_series.head())
    
    # Print shapes of training and testing sets
    print("CMCSA Feature Train/Test Shapes:", cmcsa_feature_train.shape, cmcsa_feature_test.shape)
    print("CMCSA Target Train/Test Shapes:", cmcsa_target_train.shape, cmcsa_target_test.shape)
    print("CMCSA Scaled Feature Train/Test Shapes:", cmcsa_feature_train_scaled.shape, cmcsa_feature_test_scaled.shape)
    print("CMCSA Scaled Target Train/Test Shapes:", cmcsa_target_train_scaled.shape, cmcsa_target_test_scaled.shape)
    print("CMCSA Sequence X_train_cmcsa shape:", X_train_cmcsa.shape)
    print("CMCSA Sequence y_train_cmcsa shape:", y_train_cmcsa.shape)
    print("CMCSA Sequence X_test_cmcsa shape:", X_test_cmcsa.shape)
    print("CMCSA Sequence y_test_cmcsa shape:", y_test_cmcsa.shape)

    # Initialize the Hybrid CNN-LSTM Attention model
    cmcsa_model = HybridCNN_LSTM_Attention(input_features=X_train_cmcsa.shape[2], cnn_filters=128, lstm_hidden_size=128)

    # Print model summary
    print("\nHybrid CNN-LSTM Attention Model for CMCSA:")
    print(cmcsa_model)

    # Define optimizer
    optimizer = optim.Adam(cmcsa_model.parameters(), lr=0.001)
    print("\nOptimizer:", optimizer)

    # Define Loss Function (Mean Squared Error for regression)
    criterion = nn.MSELoss()
    print("\nLoss Function:", criterion)

    # Convert numpy arrays to PyTorch tensors
    X_train_cmcsa_tensor = torch.from_numpy(X_train_cmcsa).float()
    y_train_cmcsa_tensor = torch.from_numpy(y_train_cmcsa).float() # Removed .unsqueeze(1) based on previous SPYG correction

    # Create a TensorDataset and DataLoader
    train_dataset_cmcsa = data.TensorDataset(X_train_cmcsa_tensor, y_train_cmcsa_tensor)
    train_loader_cmcsa = data.DataLoader(dataset=train_dataset_cmcsa, batch_size=32, shuffle=False)

    # Define the number of epochs
    num_epochs = 50 # Using 50 epochs as before

    # Move model to device if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cmcsa_model.to(device)
    print(f"\nTraining CMCSA model on {device}")

    # Training loop
    for epoch in range(num_epochs):
        cmcsa_model.train() # Set the model to training mode
        for batch_idx, (X_batch_cmcsa, y_batch_cmcsa) in enumerate(train_loader_cmcsa):
            X_batch_cmcsa, y_batch_cmcsa = X_batch_cmcsa.to(device), y_batch_cmcsa.to(device)

            # Forward pass
            outputs_cmcsa = cmcsa_model(X_batch_cmcsa)
            loss = criterion(outputs_cmcsa, y_batch_cmcsa)

            # Backward and optimize
            optimizer.zero_grad() # Clear gradients
            loss.backward()       # Compute gradients
            optimizer.step()      # Update weights

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("CMCSA model training complete!")

    # Evaluation
    cmcsa_model.eval()

    X_test_cmcsa_tensor = torch.from_numpy(X_test_cmcsa).float().to(device)

    with torch.no_grad():
        y_preds_cmcsa_scaled = cmcsa_model(X_test_cmcsa_tensor).cpu().numpy()

    y_preds_cmcsa = scaler_y.inverse_transform(y_preds_cmcsa_scaled)
    y_true_cmcsa = scaler_y.inverse_transform(y_test_cmcsa)

    # Calculate evaluation metrics
    mse_cmcsa = mean_squared_error(y_true_cmcsa, y_preds_cmcsa)
    rmse_cmcsa = np.sqrt(mse_cmcsa)
    mae_cmcsa = mean_absolute_error(y_true_cmcsa, y_preds_cmcsa)
    r2_cmcsa = r2_score(y_true_cmcsa, y_preds_cmcsa)

    # Print evaluation metrics
    print(f"\nCMCSA Hybrid CNN-LSTM Attention Model Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse_cmcsa:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_cmcsa:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_cmcsa:.4f}")
    print(f"R-squared (R2): {r2_cmcsa:.4f}")

    # Plot predictions
    plt.figure(figsize=(15, 7))
    plt.plot(y_true_cmcsa, label='Actual Price', color='blue')
    plt.plot(y_preds_cmcsa, label='Predicted Price', color='red', linestyle='--')
    plt.title('CMCSA Stock Price Prediction vs Actual')
    plt.xlabel('Time (Days)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("CMCSA predictions generated and plot displayed.")
    
    # Save model and scalers as production artifacts
    artifacts = ModelArtifacts(cmcsa_model, scaler_X, scaler_y, ticker='CMCSA')
    artifacts.save('C:/Users/Gyabaah/Desktop/stock_predictor/saved_models/cmcsa_artifacts.joblib')
    print("\n✓ CMCSA model and scalers saved as 'cmcsa_artifacts.joblib'. Ready for production!")

    
if __name__ == "__main__":
    main()
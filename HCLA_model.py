"""
Author: Gyabaah
Date: 12-21-2025
Description: 
A hybrid model combining CNN for feature extraction, LSTM for temporal modeling, 
and attention mechanism for sequence processing. 
This model is designed for sequence prediction tasks.
It includes a CNN layer for feature extraction, an LSTM layer for temporal modeling, 
and an attention mechanism for sequence processing. The model is trained using the Adam optimizer and 
mean squared error (MSE) loss function. The model is evaluated using the same loss function.

Returns:
    nn.module: A PyTorch model combining CNN, LSTM, and Attention Mechanism.
"""

import torch
import torch.nn as nn
import joblib


# CNN model definition for feature extraction
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels, filters=128, kernel_size=3, pool_size=2):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=input_channels, # Number of features per time step
            out_channels=filters,
            kernel_size=kernel_size,
            padding='same' # 'same' padding in Keras usually means output length is same as input length
        )
        self.relu = nn.ReLU()
        self.maxpool1d = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, x):
        # PyTorch Conv1d expects input in (batch_size, channels, sequence_length)
        # Our input X_train/X_test from create_sequences is (batch_size, sequence_length, features)
        # So we need to permute the dimensions
        x = x.permute(0, 2, 1) # Changes to (batch_size, features, sequence_length)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool1d(x)
        # Permute back if subsequent layers expect (batch_size, sequence_length, features)
        # or adjust subsequent layers to work with (batch_size, channels, sequence_length)
        # For now, let's keep it in (batch_size, features, sequence_length) after pooling
        x = x.permute(0, 2, 1) # Changes back to (batch_size, sequence_length_after_pooling, features)
        return x


# LSTM temporal model definition
class LSTMTemporalModeling(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.0, bidirectional=False):
        super(LSTMTemporalModeling, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input/output tensors are (batch, seq, feature)
            dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self, x):
        # x is expected to be (batch_size, sequence_length, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        return lstm_out


# Attention mechanism
class AttentionMechanism(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionMechanism, self).__init__()
        # Corresponds to Keras Dense(1, activation='tanh')(x)
        # Input to this layer will be (batch_size, sequence_length, feature_dim)
        # We want to apply linear transformation for each time step's feature vector.
        # Output will be (batch_size, sequence_length, 1) after linear + tanh
        self.attention_weights_layer = nn.Linear(feature_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1) # Apply softmax along the sequence dimension (axis=1)

    def forward(self, x):
        # x is expected to be (batch_size, sequence_length, feature_dim)

        # Compute attention scores (batch_size, sequence_length, 1)
        attention_scores = self.attention_weights_layer(x)
        attention_scores = self.tanh(attention_scores)

        # Normalize attention weights (batch_size, sequence_length, 1)
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights (element-wise multiplication)
        # (batch_size, sequence_length, feature_dim) * (batch_size, sequence_length, 1)
        # PyTorch broadcasting handles the last dimension.
        context_vector = x * attention_weights

        return context_vector, attention_weights
    

# Output layer
class OutputLayer(nn.Module):
    def __init__(self, input_features):
        super(OutputLayer, self).__init__()
        # The input to the final Dense layer in Keras often comes after some form of global pooling or flattening.
        # If we sum the context vector across the sequence dimension, the input features will be the feature_dim
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        # x is expected to be (batch_size, sequence_length, feature_dim) from attention
        # Sum across the sequence dimension (dim=1) to get (batch_size, feature_dim)
        # This mimics a GlobalAveragePooling1D or similar aggregation before the final Dense layer
        x = torch.sum(x, dim=1) # Summing the context vector across the sequence length
        out = self.linear(x)
        return out


# Model class combining all components
# Define the complete Hybrid CNN-LSTM Attention Model
class HybridCNN_LSTM_Attention(nn.Module):
    def __init__(self, input_features, cnn_filters=128, lstm_hidden_size=128, cnn_kernel_size=3, cnn_pool_size=2):
        super(HybridCNN_LSTM_Attention, self).__init__()
        self.cnn_extractor = CNNFeatureExtractor(
            input_channels=input_features,
            filters=cnn_filters,
            kernel_size=cnn_kernel_size,
            pool_size=cnn_pool_size
        )
        # The LSTM input size will be the output filters of the CNN
        self.lstm_temporal_modeling = LSTMTemporalModeling(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size
        )
        # The Attention mechanism's feature_dim will be the LSTM's hidden_size
        self.attention = AttentionMechanism(feature_dim=lstm_hidden_size)
        # The OutputLayer's input_features will be the LSTM's hidden_size after summing attention output
        self.output_layer = OutputLayer(input_features=lstm_hidden_size)

    def forward(self, x):
        # x should be (batch_size, sequence_length, features)
        cnn_output = self.cnn_extractor(x)
        lstm_output = self.lstm_temporal_modeling(cnn_output)
        attended_output, _ = self.attention(lstm_output)
        final_prediction = self.output_layer(attended_output)
        return final_prediction
    
    # save the model
    def save_model(self, path):
        joblib.dump(self.state_dict(), path)
        
    # save scaler
    def save_scaler(self, scaler, path):
        joblib.dump(scaler, path)
        
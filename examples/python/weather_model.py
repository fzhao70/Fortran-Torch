#!/usr/bin/env python3
"""
Weather Model Neural Network Example

This script demonstrates a more realistic example for weather modeling:
a neural network that could be used for parameterization or correction
in a weather/climate model. It takes a 3D field (e.g., temperature, humidity)
and produces corrections or parameterized values.
"""

import torch
import torch.nn as nn
import numpy as np


class WeatherParameterization(nn.Module):
    """
    A neural network for weather model parameterization.

    This network takes 3D atmospheric fields and produces parameterized
    outputs (e.g., diabatic heating, moisture tendencies).

    Input: (batch, vertical_levels, lat, lon, features)
    Output: (batch, vertical_levels, lat, lon, output_features)
    """

    def __init__(self, in_channels=5, out_channels=3, hidden_channels=16, vertical_levels=10):
        super(WeatherParameterization, self).__init__()

        self.vertical_levels = vertical_levels

        # 2D convolutions for spatial patterns
        self.conv1 = nn.Conv2d(in_channels * vertical_levels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels * vertical_levels, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(hidden_channels)
        self.batch_norm2 = nn.BatchNorm2d(hidden_channels)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, vertical_levels, features, lat, lon)
               or (batch, features) for flattened input

        Returns:
            Output tensor of shape (batch, vertical_levels, out_features, lat, lon)
            or (batch, output_features) for flattened output
        """
        # Handle both flattened and spatial inputs
        if x.dim() == 2:
            # Flattened input: (batch, features) -> unflatten to spatial
            batch_size = x.shape[0]
            # Assume square spatial dimensions
            total_features = x.shape[1]
            # For this example, reshape to a reasonable spatial size
            spatial_size = 8  # 8x8 grid
            features_per_level = total_features // (self.vertical_levels * spatial_size * spatial_size)

            x = x.view(batch_size, self.vertical_levels, features_per_level, spatial_size, spatial_size)

        batch_size, vlevels, features, lat, lon = x.shape

        # Reshape to combine vertical levels with features
        x = x.view(batch_size, vlevels * features, lat, lon)

        # Apply convolutions
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        x = self.conv3(x)

        # Reshape back to separate vertical levels
        out_features = x.shape[1] // vlevels
        x = x.view(batch_size, vlevels, out_features, lat, lon)

        return x


class SimplifiedWeatherNet(nn.Module):
    """
    Simplified version that works with 1D or 2D inputs.
    Useful for testing with Fortran.
    """

    def __init__(self, input_size=50, output_size=20):
        super(SimplifiedWeatherNet, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, input_size)

        Returns:
            Output tensor of shape (batch, output_size)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x


def train_weather_model():
    """Train a simplified weather parameterization model."""
    print("Creating weather parameterization neural network...")

    # Model parameters
    input_size = 50   # e.g., temperature, humidity, wind at multiple levels
    output_size = 20  # e.g., heating rates, moisture tendencies
    batch_size = 64
    epochs = 200

    # Create model
    model = SimplifiedWeatherNet(input_size, output_size)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Generate synthetic training data
    # In reality, this would be from actual weather model data
    print("Generating synthetic atmospheric data...")
    n_samples = 5000

    # Input: atmospheric state variables
    X_train = torch.randn(n_samples, input_size)

    # Output: parameterized tendencies (simplified)
    # In reality, these would be computed from high-resolution simulations
    y_train = torch.randn(n_samples, output_size)

    # Split into train and validation
    split = int(0.8 * n_samples)
    X_train, X_val = X_train[:split], X_train[split:]
    y_train, y_val = y_train[:split], y_train[split:]

    # Training loop
    print("Training model...")
    model.train()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

        scheduler.step(val_loss)
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss/len(X_train)*batch_size:.4f}, '
                  f'Val Loss: {val_loss:.4f}')

    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    return model


def export_model(model, filename='weather_model.pt'):
    """Export the model to TorchScript format."""
    print(f"\nExporting model to {filename}...")

    model.eval()

    # Create example input for a weather model
    # Shape: (batch, input_features)
    example_input = torch.randn(1, 50)

    # Trace the model
    traced_model = torch.jit.trace(model, example_input)

    # Save the traced model
    traced_model.save(filename)
    print(f"Model exported successfully!")

    # Verify the model
    print("\nVerifying exported model...")
    loaded_model = torch.jit.load(filename)

    test_input = torch.randn(10, 50)  # Batch of 10

    with torch.no_grad():
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)

        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {original_output.shape}")
        print(f"Outputs match: {torch.allclose(original_output, loaded_output, rtol=1e-5)}")

    print("\nModel is ready for integration with Fortran weather models!")
    print(f"Input: (batch_size, 50) - atmospheric state variables")
    print(f"Output: (batch_size, 20) - parameterized tendencies")
    print("\nExample usage in weather model:")
    print("1. Extract atmospheric state at each grid point/time step")
    print("2. Call the neural network for parameterization")
    print("3. Apply the predicted tendencies to the model state")


if __name__ == '__main__':
    # Train the model
    model = train_weather_model()

    # Export to TorchScript
    export_model(model)

    print("\n" + "="*60)
    print("INTEGRATION NOTES FOR WEATHER MODELS")
    print("="*60)
    print("""
This neural network can be integrated into a Fortran-based weather model as:

1. Physics Parameterization:
   - Replace or augment existing parameterization schemes
   - Input: T, q, u, v at multiple levels
   - Output: Heating rates, moisture tendencies

2. Bias Correction:
   - Correct systematic model biases
   - Input: Model state variables
   - Output: Corrections to apply

3. Downscaling:
   - Predict high-resolution features from coarse input
   - Input: Coarse-resolution fields
   - Output: High-resolution corrections

4. Data Assimilation:
   - Learn observation operators
   - Input: Model state
   - Output: Predicted observations

Key considerations:
- Ensure input normalization matches training data
- Handle vertical levels consistently
- Consider computational cost in operational settings
- Validate extensively against traditional methods
    """)

#!/usr/bin/env python3
"""
Simple Neural Network Example

This script demonstrates how to create, train, and export a simple PyTorch model
that can be used with the Fortran-Torch framework.
"""

import torch
import torch.nn as nn
import numpy as np


class SimpleNet(nn.Module):
    """A simple feedforward neural network."""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def train_model():
    """Train the model on synthetic data."""
    print("Creating simple neural network...")

    # Model parameters
    input_size = 10
    hidden_size = 20
    output_size = 5
    batch_size = 32
    epochs = 100

    # Create model
    model = SimpleNet(input_size, hidden_size, output_size)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Generate synthetic training data
    print("Generating synthetic training data...")
    X_train = torch.randn(1000, input_size)
    # Simple target: sum of inputs passed through a transformation
    y_train = torch.randn(1000, output_size)

    # Training loop
    print("Training model...")
    model.train()
    for epoch in range(epochs):
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print("Training completed!")
    return model


def export_model(model, filename='simple_model.pt'):
    """Export the model to TorchScript format."""
    print(f"\nExporting model to {filename}...")

    # Set model to evaluation mode
    model.eval()

    # Create example input
    example_input = torch.randn(1, 10)

    # Trace the model
    traced_model = torch.jit.trace(model, example_input)

    # Save the traced model
    traced_model.save(filename)
    print(f"Model exported successfully!")

    # Verify the model can be loaded
    print("\nVerifying exported model...")
    loaded_model = torch.jit.load(filename)
    test_input = torch.randn(1, 10)

    with torch.no_grad():
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)

        print(f"Original output shape: {original_output.shape}")
        print(f"Loaded output shape: {loaded_output.shape}")
        print(f"Outputs match: {torch.allclose(original_output, loaded_output)}")

    print("\nModel is ready to use with Fortran-Torch!")
    print(f"Input shape: (batch_size, 10)")
    print(f"Output shape: (batch_size, 5)")


if __name__ == '__main__':
    # Train the model
    model = train_model()

    # Export to TorchScript
    export_model(model)

#!/usr/bin/env python3
"""
Create a simple test model for unit testing.

This creates a minimal TorchScript model that can be used
for testing the Fortran-Torch framework.
"""

import torch
import torch.nn as nn


class TestModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


def create_test_model(output_path='test_model.pt'):
    """Create and save a test model."""
    print("Creating test model...")

    # Create model
    model = TestModel()

    # Initialize with known weights for reproducibility
    torch.manual_seed(42)
    with torch.no_grad():
        model.fc.weight.fill_(0.1)
        model.fc.bias.fill_(0.0)

    # Set to eval mode
    model.eval()

    # Create example input
    example_input = torch.randn(1, 10)

    # Trace the model
    traced_model = torch.jit.trace(model, example_input)

    # Save
    traced_model.save(output_path)

    print(f"Test model saved to: {output_path}")

    # Verify
    loaded = torch.jit.load(output_path)
    test_input = torch.randn(1, 10)
    output = loaded(test_input)

    print(f"  Input shape:  {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print("  Model verified successfully!")

    return output_path


if __name__ == '__main__':
    import sys
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'test_model.pt'
    create_test_model(output_path)

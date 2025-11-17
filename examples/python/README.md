# Python Examples

This directory contains Python scripts for training and exporting PyTorch models to TorchScript format for use with Fortran-Torch.

## Setup

Install required Python packages:

```bash
pip install -r ../../requirements.txt
```

## Examples

### 1. Simple Model (`simple_model.py`)

A basic feedforward neural network demonstrating the workflow:

```bash
python simple_model.py
```

**Output**: `simple_model.pt` - A TorchScript model file

**Specifications**:
- Input: (batch_size, 10) - 10 features
- Output: (batch_size, 5) - 5 outputs
- Architecture: 2 hidden layers with ReLU activation

### 2. Weather Model (`weather_model.py`)

A more realistic example for atmospheric science applications:

```bash
python weather_model.py
```

**Output**: `weather_model.pt` - A TorchScript model file

**Specifications**:
- Input: (batch_size, 50) - Atmospheric state variables
- Output: (batch_size, 20) - Parameterized tendencies
- Architecture: 3 fully-connected layers with dropout

## Model Export Checklist

When exporting your own models for Fortran-Torch:

- [ ] Set model to evaluation mode: `model.eval()`
- [ ] Use `torch.jit.trace()` or `torch.jit.script()` for export
- [ ] Test the exported model loads correctly
- [ ] Verify outputs match between original and exported model
- [ ] Document input/output shapes and data types
- [ ] Consider normalization (include in model if possible)
- [ ] Test with representative input data

## Creating Your Own Model

Template for exporting any PyTorch model:

```python
import torch
import torch.nn as nn

# 1. Define your model
class YourModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        # ... more layers ...

    def forward(self, x):
        # Define forward pass
        x = self.layer1(x)
        # ... more operations ...
        return x

# 2. Train your model
model = YourModel()
# ... training code ...

# 3. Export to TorchScript
model.eval()
example_input = torch.randn(1, input_size)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('your_model.pt')

# 4. Verify
loaded = torch.jit.load('your_model.pt')
test_input = torch.randn(10, input_size)
output = loaded(test_input)
print(f"Output shape: {output.shape}")
```

## Tips for Weather/Climate Models

When creating ML models for weather and climate applications:

1. **Input Features**: Typical inputs include
   - Temperature, humidity, wind components at multiple levels
   - Surface variables (pressure, precipitation)
   - Derived quantities (stability indices, etc.)

2. **Output Features**: Common outputs
   - Heating rates (K/s or K/day)
   - Moisture tendencies (kg/kg/s)
   - Momentum tendencies (m/s/s)
   - Radiative fluxes (W/m²)

3. **Normalization**: Essential for ML
   - Normalize inputs to similar scales
   - Document normalization parameters
   - Include normalization in the model when possible

4. **Validation**: Critical for operational use
   - Compare against physics-based parameterizations
   - Test across different atmospheric conditions
   - Ensure physical consistency (conservation laws)
   - Validate energy and mass budgets

5. **Performance**: Consider computational cost
   - Profile inference time
   - Compare to traditional parameterizations
   - Consider model size vs accuracy tradeoff

# Integrating Fortran-Torch into Weather and Climate Models

Comprehensive guide for integrating PyTorch ML models into operational weather and climate models using Fortran-Torch.

## Overview

This document provides guidance for integrating Fortran-Torch into major weather and climate modeling systems. For model-specific details, see:

- **[WRF Integration Guide](INTEGRATION_WRF.md)** - Weather Research and Forecasting Model
- **[MPAS Integration Guide](INTEGRATION_MPAS.md)** - Model for Prediction Across Scales
- **[FV3 Integration Guide](INTEGRATION_FV3.md)** - FV3GFS Operational Global Model

## Supported Models

| Model | Type | Grid | Physics | Status | Guide |
|-------|------|------|---------|--------|-------|
| WRF | Regional | Regular lat-lon | Modular | ✓ Tested | [WRF](INTEGRATION_WRF.md) |
| MPAS | Global/Regional | Unstructured Voronoi | Column | ✓ Tested | [MPAS](INTEGRATION_MPAS.md) |
| FV3 | Global | Cubed-sphere | CCPP | ✓ Tested | [FV3](INTEGRATION_FV3.md) |
| CAM | Global | Spectral/FV | CESM | ⚠ Untested | Contact us |
| UM | Global | Regular lat-lon | Unified | ⚠ Untested | Contact us |

## Common Integration Patterns

### 1. Physics Parameterization

**Use Case**: Replace or augment physics schemes

**Example Applications**:
- Deep convection
- Planetary boundary layer
- Cloud microphysics
- Radiation corrections
- Gravity wave drag

**Integration Pattern**:

```fortran
module ml_physics_scheme

    use ftorch
    use model_constants  ! Model-specific

    type(torch_model) :: ml_model

contains

subroutine ml_physics_init(model_path)
    ! Load model once at initialization
    ml_model = torch_load_model(model_path, TORCH_DEVICE_CPU)
end subroutine

subroutine ml_physics_driver(state, tendencies, grid_info)
    ! Called each physics timestep
    ! Extract state → Run ML → Apply tendencies
end subroutine

end module ml_physics_scheme
```

**Key Considerations**:
- Extract atmospheric state to ML format
- Handle grid-specific information (resolution, tile, etc.)
- Convert ML output to model tendencies
- Ensure conservative properties (mass, energy)

### 2. Bias Correction

**Use Case**: Correct systematic model errors

**Example Applications**:
- Temperature/humidity bias
- Precipitation correction
- Surface variable adjustment
- Forecast error reduction

**Integration Pattern**:

```fortran
subroutine apply_ml_bias_correction(forecast, location_info)

    ! Extract forecast variables
    ! Include metadata (time, location, season)
    ! Run ML correction model
    ! Apply corrections to forecast fields

end subroutine
```

**Key Considerations**:
- Train on model-analysis differences
- Include temporal/spatial context
- Preserve physical constraints
- Handle extreme events

### 3. Subgrid Parameterization

**Use Case**: Parameterize unresolved processes

**Example Applications**:
- Turbulent fluxes
- Subgrid clouds
- Orographic effects
- Urban canopy

**Integration Pattern**:

```fortran
subroutine ml_subgrid_scheme(resolved_state, subgrid_tendencies, resolution)

    ! Key: Include resolution as input for scale-aware ML
    ! ML learns different behavior at different scales

    call extract_resolved_state(resolved_state, ml_input)
    ml_input(n_features) = resolution  ! Scale information
    output_tensor = torch_forward(model, input_tensor)

end subroutine
```

**Key Considerations**:
- Scale-aware training (variable resolution)
- Respect resolved vs unresolved separation
- Ensure proper scaling with resolution

### 4. Data Assimilation

**Use Case**: ML-enhanced observation operators

**Example Applications**:
- Satellite radiance operators
- Radar reflectivity operators
- GPS radio occultation
- Feature extraction

**Integration Pattern**:

```fortran
subroutine ml_observation_operator(model_state, obs_type, predicted_obs)

    ! Forward operator: state → observations
    ! More accurate than traditional radiative transfer for complex obs

end subroutine
```

### 5. Post-Processing

**Use Case**: Enhance model output

**Example Applications**:
- Statistical downscaling
- Ensemble calibration
- Derived products
- Uncertainty quantification

**Integration Pattern**:

```fortran
subroutine ml_postprocess(raw_forecast, enhanced_output)

    ! Applied to model output
    ! No impact on model evolution
    ! Can be more experimental

end subroutine
```

## General Integration Workflow

### Step 1: Build Fortran-Torch

```bash
# Download LibTorch
cd Fortran-Torch
./scripts/download_libtorch.sh cpu  # or cu118 for GPU

# Build (match your model's compiler!)
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_Fortran_COMPILER=gfortran \  # or ifort
      -DCMAKE_CXX_COMPILER=g++ \           # or icpc
      -DCMAKE_INSTALL_PREFIX=/opt/fortran-torch \
      ..
make -j8
make install
```

### Step 2: Train and Export ML Model

**Python Side**:

```python
import torch
import torch.nn as nn

# 1. Define model
class WeatherML(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture

    def forward(self, x):
        # Forward pass
        return output

# 2. Train
model = WeatherML()
# ... training code ...

# 3. Export to TorchScript
model.eval()
example_input = torch.randn(1, input_size)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('weather_ml.pt')

# 4. Verify
loaded = torch.jit.load('weather_ml.pt')
test_output = loaded(example_input)
print(f"Model ready: {test_output.shape}")
```

### Step 3: Create Fortran Module

**Fortran Side**:

```fortran
module ml_integration

    use ftorch
    use iso_fortran_env, only: real32

    type(torch_model) :: ml_model
    logical :: initialized = .false.

contains

subroutine ml_init(model_path)
    character(len=*), intent(in) :: model_path

    ml_model = torch_load_model(trim(model_path), TORCH_DEVICE_CPU)
    if (c_associated(ml_model%ptr)) then
        initialized = .true.
        print *, 'ML model loaded successfully'
    else
        print *, 'ERROR: Failed to load ML model'
    end if
end subroutine

subroutine ml_run(input_data, output_data)
    real(real32), intent(in) :: input_data(:)
    real(real32), intent(out) :: output_data(:)

    type(torch_tensor) :: input_tensor, output_tensor

    if (.not. initialized) return

    input_tensor = torch_tensor_from_array(input_data)
    output_tensor = torch_forward(ml_model, input_tensor)
    call torch_tensor_to_array(output_tensor, output_data)

    call torch_free_tensor(input_tensor)
    call torch_free_tensor(output_tensor)
end subroutine

subroutine ml_finalize()
    if (initialized) then
        call torch_free_model(ml_model)
        initialized = .false.
    end if
end subroutine

end module ml_integration
```

### Step 4: Integrate into Model

1. **Modify Build System**: Add Fortran-Torch libraries
2. **Call Initialization**: Load model at model startup
3. **Call in Physics Loop**: Run ML during physics
4. **Call Finalization**: Clean up at model end

### Step 5: Test and Validate

```bash
# 1. Unit test ML module
./test_ml_module

# 2. Run short forecast
./model.exe namelist.test

# 3. Compare with baseline
python verify_forecast.py

# 4. Long-term testing
./run_regression_tests.sh
```

## Grid-Specific Considerations

### Regular Grids (WRF, CAM-FV, UM)

```fortran
! Straightforward column processing
do j = jstart, jend
    do i = istart, iend
        call extract_column(i, j, state, ml_input)
        call ml_forward(ml_input, ml_output)
        call apply_tendencies(i, j, ml_output, tendencies)
    end do
end do
```

### Unstructured Grids (MPAS, ICON)

```fortran
! Cell-based processing, include resolution
do iCell = 1, nCells
    resolution = get_cell_resolution(iCell)
    call extract_column(iCell, state, ml_input)
    ml_input(n_features) = resolution  ! Important!
    call ml_forward(ml_input, ml_output)
    call apply_tendencies(iCell, ml_output, tendencies)
end do
```

### Cubed-Sphere (FV3, CAM-SE)

```fortran
! Tile-based processing
do iTile = 1, 6
    do j = 1, ny
        do i = 1, nx
            call extract_column(iTile, i, j, state, ml_input)
            call ml_forward(ml_input, ml_output)
            call apply_tendencies(iTile, i, j, ml_output, tendencies)
        end do
    end do
end do
```

### Spectral Models

```fortran
! Transform to gridpoint, apply ML, transform back
call spectral_to_grid(state_spectral, state_grid)

do iPoint = 1, nPoints
    call ml_process(state_grid(iPoint), tendencies_grid(iPoint))
end do

call grid_to_spectral(tendencies_grid, tendencies_spectral)
```

## Performance Optimization

### 1. Minimize Data Transfer

```fortran
! Bad: Create/destroy tensors in tight loop
do i = 1, nx
    tensor = torch_tensor_from_array(data(i,:))  ! Slow
    output = torch_forward(model, tensor)
    call torch_free_tensor(tensor)
end do

! Good: Batch or reuse
call batch_inference(data, nx, outputs)
```

### 2. Thread Safety

```fortran
! OpenMP-safe pattern
!$OMP PARALLEL DO PRIVATE(ml_input, ml_output, input_tensor, output_tensor)
do i = 1, n
    ! Each thread needs its own tensors
    input_tensor = torch_tensor_from_array(ml_input)
    output_tensor = torch_forward(model, input_tensor)
    call torch_free_tensor(input_tensor)
    call torch_free_tensor(output_tensor)
end do
!$OMP END PARALLEL DO
```

### 3. MPI Considerations

```fortran
! Each MPI rank loads its own model (no communication needed)
call mpi_comm_rank(comm, myrank, ierr)

! Load model on each rank
model = torch_load_model('model.pt', TORCH_DEVICE_CPU)

! No MPI communication needed during inference
! Each rank processes its own domain
```

### 4. Memory Management

```fortran
! Monitor memory usage
call get_memory_usage(rss_mb)

if (rss_mb > threshold) then
    write(*,*) 'WARNING: High memory usage:', rss_mb, 'MB'
end if

! Clean up promptly
call torch_free_tensor(tensor)  ! Don't delay
```

## Validation Strategies

### 1. Single-Column Tests

```fortran
! Test ML scheme on single column
call run_single_column_test(input_profile, ml_output)
call compare_with_baseline(ml_output, expected_output)
```

### 2. Idealized Cases

- WRF: Quarter-circle supercell
- MPAS: Jablonowski-Williamson baroclinic wave
- FV3: Held-Suarez test

### 3. Real-Data Cases

- Historical events (hurricanes, heavy precipitation)
- Seasonal verification
- Climate statistics

### 4. Ensemble Verification

```python
# Compare ensemble spread
import numpy as np

baseline_spread = np.std(baseline_ensemble, axis=0)
ml_spread = np.std(ml_ensemble, axis=0)

# ML should maintain or improve spread
print(f'Spread ratio: {ml_spread.mean() / baseline_spread.mean():.2f}')
```

## Common Issues and Solutions

### Issue: Model fails to load

```fortran
! Solution: Check path and verify file
inquire(file=trim(model_path), exist=file_exists)
if (.not. file_exists) then
    write(*,*) 'ERROR: Model file not found:', trim(model_path)
end if
```

### Issue: Segmentation fault

```fortran
! Solution: Check pointer associations
if (.not. c_associated(tensor%ptr)) then
    write(*,*) 'ERROR: Tensor not created'
    return
end if
```

### Issue: Poor scaling with MPI

```bash
# Each rank loads model independently
# No cross-rank communication needed for inference
# Profile to identify bottlenecks
```

### Issue: Memory leak

```fortran
! Solution: Always free tensors and models
call torch_free_tensor(input_tensor)
call torch_free_tensor(output_tensor)
! In finalization:
call torch_free_model(ml_model)
```

## Best Practices

### Development

1. **Start Simple**: Begin with post-processing, then physics
2. **Test Thoroughly**: Unit tests → idealized → real cases
3. **Version Control**: Track model versions
4. **Documentation**: Document assumptions and limitations
5. **Monitoring**: Log performance and errors

### Training

1. **Use Model Data**: Train on actual model output
2. **Match Precision**: Train with same precision as model
3. **Include Metadata**: Resolution, location, time, etc.
4. **Validate Broadly**: Test across seasons, regions, conditions
5. **Conservative Properties**: Ensure conservation if needed

### Deployment

1. **Fail-Safe**: Always have fallback to traditional physics
2. **Monitoring**: Track ML performance vs traditional
3. **Version Management**: Clear model versioning system
4. **Testing**: Extensive regression testing
5. **Documentation**: Operational procedures

## Quick Reference

### Fortran-Torch API

```fortran
! Initialization
model = torch_load_model(path, device)

! Create tensor
tensor = torch_tensor_from_array(fortran_array)

! Inference
output = torch_forward(model, input)

! Extract data
call torch_tensor_to_array(tensor, fortran_array)

! Cleanup
call torch_free_tensor(tensor)
call torch_free_model(model)

! Utilities
available = torch_cuda_available()
```

### Common Patterns

```fortran
! Pattern 1: Simple inference
input_tensor = torch_tensor_from_array(input_data)
output_tensor = torch_forward(model, input_tensor)
call torch_tensor_to_array(output_tensor, output_data)
call torch_free_tensor(input_tensor)
call torch_free_tensor(output_tensor)

! Pattern 2: Batch processing
do i = 1, n_columns
    ! Process multiple columns
end do

! Pattern 3: Conditional ML
if (use_ml_scheme) then
    call ml_physics(...)
else
    call traditional_physics(...)
end if
```

## Model-Specific Guides

For detailed integration instructions:

- **WRF**: See [INTEGRATION_WRF.md](INTEGRATION_WRF.md)
  - Regular grid
  - Namelist configuration
  - Registry modifications
  - Physics driver integration

- **MPAS**: See [INTEGRATION_MPAS.md](INTEGRATION_MPAS.md)
  - Unstructured mesh
  - Variable resolution
  - Cell-based processing
  - MPAS framework integration

- **FV3**: See [INTEGRATION_FV3.md](INTEGRATION_FV3.md)
  - Cubed-sphere grid
  - CCPP compliance
  - Operational considerations
  - Suite configuration

## Support and Resources

### Documentation
- [Fortran-Torch README](../README.md)
- [API Reference](../README.md#api-reference)
- [Examples](../examples/)
- [Testing Guide](../TESTING.md)

### Community
- GitHub Issues: Report bugs and issues
- Discussions: Ask questions
- Contributing: See [CONTRIBUTING.md](../CONTRIBUTING.md)

### Training Materials
- Example notebooks (coming soon)
- Tutorial videos (coming soon)
- Workshop materials (coming soon)

## Citation

If you use Fortran-Torch in your research, please cite:

```bibtex
@software{fortran_torch,
  title = {Fortran-Torch: PyTorch Integration for Fortran-based Models},
  author = {Fanghe Zhao},
  year = {2024},
  url = {https://github.com/fzhao70/Fortran-Torch}
}
```

## Roadmap

Future enhancements:

- [ ] Additional model integrations (CAM, UM, etc.)
- [ ] GPU multi-node support
- [ ] Automated testing frameworks
- [ ] Benchmarking suite
- [ ] Online learning capabilities
- [ ] Model interpretability tools

---

**Questions?** Open an issue on [GitHub](https://github.com/fzhao70/Fortran-Torch/issues) or consult the model-specific guides.

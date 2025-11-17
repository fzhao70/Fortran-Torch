# Fortran-Torch Documentation

This directory contains comprehensive integration guides for using Fortran-Torch with major weather and climate models.

## Documentation Index

### Weather Model Integration Guides

| Document | Description | Model | Grid Type |
|----------|-------------|-------|-----------|
| [Weather Model Integration](WEATHER_MODEL_INTEGRATION.md) | **Master guide** - General patterns, best practices, common use cases | All | All |
| [WRF Integration](INTEGRATION_WRF.md) | WRF-specific integration guide | WRF v4.x+ | Regular lat-lon |
| [MPAS Integration](INTEGRATION_MPAS.md) | MPAS-specific integration guide | MPAS v7.x+ | Unstructured Voronoi |
| [FV3 Integration](INTEGRATION_FV3.md) | FV3GFS-specific integration guide | FV3 v16+ | Cubed-sphere |

### Quick Reference

**New to weather model integration?** Start here:
1. Read [Weather Model Integration](WEATHER_MODEL_INTEGRATION.md) for overview
2. Find your model's specific guide (WRF/MPAS/FV3)
3. Follow the step-by-step integration instructions
4. See [../examples/](../examples/) for working examples

## Guide Contents

### [Weather Model Integration](WEATHER_MODEL_INTEGRATION.md)

**Purpose**: Comprehensive guide for all weather models

**Covers**:
- Common integration patterns (physics, bias correction, post-processing)
- General workflow (build, train, integrate, test)
- Grid-specific considerations (regular, unstructured, cubed-sphere, spectral)
- Performance optimization
- Validation strategies
- Best practices

**Audience**: Anyone integrating ML into weather models

### [WRF Integration](INTEGRATION_WRF.md)

**Purpose**: Detailed WRF-specific integration

**Covers**:
- WRF build system modifications
- Physics module creation (`module_cu_ml.F`, `module_bl_ml.F`)
- Registry variable registration
- Namelist configuration
- OpenMP threading considerations
- MPI deployment
- Testing with WRF test cases

**Key Features**:
- Complete ML convection scheme example
- PBL parameterization example
- Microphysics enhancement example
- Performance profiling
- Troubleshooting guide

**Audience**: WRF users and developers

### [MPAS Integration](INTEGRATION_MPAS.md)

**Purpose**: Detailed MPAS-specific integration

**Covers**:
- MPAS build system modifications
- Unstructured mesh handling
- Variable resolution considerations
- Scale-aware ML models
- Cell-based processing
- MPAS framework integration
- Registry.xml modifications

**Key Features**:
- ML convection on unstructured mesh
- Subgrid turbulence parameterization
- Resolution-aware training examples
- Mesh connectivity handling
- Load balancing for unstructured grids

**Unique Aspects**:
- Variable resolution ML (essential for MPAS)
- Voronoi mesh processing
- Cell neighbor information

**Audience**: MPAS users and developers

### [FV3 Integration](INTEGRATION_FV3.md)

**Purpose**: Detailed FV3GFS-specific integration

**Covers**:
- CCPP-compliant scheme development
- CCPP metadata creation (`.meta` files)
- Physics suite configuration
- Cubed-sphere grid handling
- Operational deployment considerations
- Fail-safe mechanisms
- Performance monitoring

**Key Features**:
- Full CCPP-compliant ML scheme
- CCPP metadata examples
- Suite XML configuration
- Tile-based processing
- Operational best practices

**Unique Aspects**:
- CCPP framework compliance
- Operational readiness
- Cubed-sphere specifics
- NOAA/EMC standards

**Audience**: FV3 users, NOAA/EMC developers, operational centers

## Integration Comparison

| Feature | WRF | MPAS | FV3 |
|---------|-----|------|-----|
| **Grid** | Regular lat-lon | Unstructured Voronoi | Cubed-sphere |
| **Resolution** | Fixed | Variable | Fixed per tile |
| **Physics** | Modular schemes | Column physics | CCPP framework |
| **Complexity** | ⭐⭐ Medium | ⭐⭐⭐ High | ⭐⭐⭐ High |
| **ML Integration** | Direct module | Resolution-aware | CCPP-compliant |
| **Best For** | Regional forecasting | Global variable-res | Operational global |

## Common Integration Steps

All models follow similar high-level steps:

1. **Build Fortran-Torch**
   ```bash
   ./scripts/download_libtorch.sh cpu
   ./scripts/build.sh
   ```

2. **Train ML Model (Python)**
   ```python
   model.eval()
   traced = torch.jit.trace(model, example)
   traced.save('model.pt')
   ```

3. **Create Fortran Module**
   ```fortran
   module ml_physics
       use ftorch
       ! ... integration code ...
   end module
   ```

4. **Modify Build System**
   - Add Fortran-Torch libraries
   - Link with LibTorch

5. **Register Variables**
   - WRF: `Registry.EM_COMMON`
   - MPAS: `Registry.xml`
   - FV3: CCPP metadata

6. **Test and Validate**
   - Unit tests
   - Idealized cases
   - Real data cases

## Use Case Examples

### Physics Parameterization

**All Models**: Replace or augment physics schemes

- **WRF**: `module_cu_ml.F` for convection
- **MPAS**: `mpas_atmphys_ml_convection.F` with resolution
- **FV3**: CCPP-compliant `ml_convection.F90`

### Bias Correction

**All Models**: Correct systematic errors

- Post-process model output
- Apply corrections before output
- Model-specific bias patterns

### Subgrid Processes

**All Models**: Parameterize unresolved scales

- **WRF**: Fixed resolution, standard approach
- **MPAS**: Scale-aware (critical for variable resolution!)
- **FV3**: Cubed-sphere aware

## Grid-Specific Patterns

### Regular Grids (WRF)

```fortran
do j = jstart, jend
    do i = istart, iend
        call ml_process_column(i, j, state, output)
    end do
end do
```

### Unstructured (MPAS)

```fortran
do iCell = 1, nCells
    resolution = get_resolution(iCell)  ! Important!
    call ml_process_column(iCell, state, resolution, output)
end do
```

### Cubed-Sphere (FV3)

```fortran
do iTile = 1, 6
    do j = 1, ny
        do i = 1, nx
            call ml_process_column(iTile, i, j, state, output)
        end do
    end do
end do
```

## Performance Tips

### All Models

1. **Minimize Tensor Creation**
   - Batch processing where possible
   - Reuse tensors

2. **Thread Safety**
   - Each OpenMP thread needs own tensors
   - Models can be shared (read-only)

3. **MPI**
   - Each rank loads its own model
   - No cross-rank tensor sharing needed

### Model-Specific

- **WRF**: OpenMP tile-based parallelization
- **MPAS**: Load balance across irregular decomposition
- **FV3**: CCPP threading considerations

## Testing

### Idealized Tests

- **WRF**: em_quarter_ss (supercell)
- **MPAS**: jw_baroclinic_wave
- **FV3**: Held-Suarez

### Real Data

All models support real-data cases for validation

## Contributing

Want to add support for other models (CAM, UM, ICON)?

1. See [CONTRIBUTING.md](../CONTRIBUTING.md)
2. Follow the integration guide template
3. Provide working example
4. Submit pull request

## Support

- **General Questions**: [GitHub Discussions](https://github.com/yourusername/Fortran-Torch/discussions)
- **Issues**: [GitHub Issues](https://github.com/yourusername/Fortran-Torch/issues)
- **Model-Specific**: Consult model community forums

## Additional Resources

### Fortran-Torch Documentation

- [Main README](../README.md) - Project overview
- [Installation Guide](../INSTALL.md) - Detailed installation
- [Architecture](../ARCHITECTURE.md) - Design details
- [Testing Guide](../TESTING.md) - Testing procedures
- [Examples](../examples/) - Working examples

### Model Documentation

- [WRF User Guide](https://www2.mmm.ucar.edu/wrf/users/)
- [MPAS Documentation](https://mpas-dev.github.io/)
- [FV3 Documentation](https://noaa-emc.github.io/FV3_Dycore_ufs-v2.0.0/html/)
- [CCPP Documentation](https://ccpp-techdoc.readthedocs.io/)

### Machine Learning Resources

- [PyTorch TorchScript](https://pytorch.org/docs/stable/jit.html)
- [LibTorch C++ API](https://pytorch.org/cppdocs/)

## Citation

If you use these integration guides in your research:

```bibtex
@software{fortran_torch_integration,
  title = {Fortran-Torch Weather Model Integration Guides},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Fortran-Torch}
}
```

---

**Questions?** Open an issue or see the model-specific guides for detailed help.

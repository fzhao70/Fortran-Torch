# Integrating Fortran-Torch into FV3GFS (Finite-Volume Cubed-Sphere Global Forecast System)

This guide shows how to integrate PyTorch machine learning models into the FV3GFS model using Fortran-Torch.

## Table of Contents

- [Overview](#overview)
- [FV3 Architecture](#fv3-architecture)
- [Build Configuration](#build-configuration)
- [Integration Points](#integration-points)
- [Example: CCPP-Compliant ML Physics](#example-ccpp-compliant-ml-physics)
- [Example: ML Post-Processing](#example-ml-post-processing)
- [Example: Data Assimilation Enhancement](#example-data-assimilation-enhancement)
- [Cubed-Sphere Grid Considerations](#cubed-sphere-grid-considerations)
- [Operational Deployment](#operational-deployment)
- [Testing and Validation](#testing-and-validation)

## Overview

FV3GFS is NOAA's operational global forecast system using a cubed-sphere grid and the Common Community Physics Package (CCPP). Fortran-Torch enables ML integration for:

- **Physics Parameterizations**: CCPP-compliant ML schemes
- **Bias Correction**: Systematic error correction
- **Post-Processing**: ML-enhanced output products
- **Data Assimilation**: Neural network observation operators
- **Hybrid Schemes**: ML augmentation of existing physics

### FV3 Repository
- **GitHub**: https://github.com/NOAA-EMC/fv3gfs
- **CCPP Physics**: https://github.com/NCAR/ccpp-physics
- **Version**: Compatible with FV3GFS v16+

## FV3 Architecture

FV3GFS has a modular structure:

```
fv3gfs/
├── FV3/                      # Dynamical core
│   ├── atmos_cubed_sphere/   # Cubed-sphere dynamics
│   └── ccpp/                 # CCPP framework
├── ccpp-physics/             # Physics parameterizations
│   ├── physics/              # Individual schemes
│   ├── GFS_layer/            # GFS-specific layers
│   └── ccpp_schemes.xml     # Scheme metadata
└── tests/                    # Test cases
```

### CCPP Structure

CCPP (Common Community Physics Package) provides a framework for physics:

- **Schemes**: Individual parameterizations
- **Suites**: Collections of schemes
- **Metadata**: XML descriptions
- **Caps**: Interface layers

## Build Configuration

### 1. Build Fortran-Torch

```bash
# Build Fortran-Torch for FV3
cd /path/to/Fortran-Torch
./scripts/download_libtorch.sh cpu  # or cu118 for GPU

mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_Fortran_COMPILER=ifort \  # FV3 often uses Intel
      -DCMAKE_CXX_COMPILER=icpc \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/apps/fortran-torch \
      ..
make -j16
make install
```

### 2. Modify FV3 Build System

FV3 uses CMake. **Edit `CMakeLists.txt`**:

```cmake
# Add Fortran-Torch paths
set(FTORCH_DIR "/apps/fortran-torch")
set(LIBTORCH_DIR "/path/to/libtorch")

# Add include directories
include_directories(${FTORCH_DIR}/include/fortran)
include_directories(${LIBTORCH_DIR}/include)

# Link libraries
target_link_libraries(fv3atm
    ${FTORCH_DIR}/lib/libftorch.so
    ${FTORCH_DIR}/lib/libfortran_torch_cpp.so
    ${LIBTORCH_DIR}/lib/libtorch.so
    ${LIBTORCH_DIR}/lib/libtorch_cpu.so
    ${LIBTORCH_DIR}/lib/libc10.so
)

# Set RPATH
set_target_properties(fv3atm PROPERTIES
    INSTALL_RPATH "${LIBTORCH_DIR}/lib"
)
```

**For CCPP Physics** - Edit `ccpp-physics/CMakeLists.txt`:

```cmake
# Add Fortran-Torch for physics
include_directories(${FTORCH_DIR}/include/fortran)

# Link to CCPP physics library
target_link_libraries(ccppphys
    ${FTORCH_DIR}/lib/libftorch.so
    ${FTORCH_DIR}/lib/libfortran_torch_cpp.so
)
```

### 3. Build FV3

```bash
cd fv3gfs
mkdir build && cd build

# Configure
cmake .. \
    -DCCPP_SUITES=FV3_GFS_v16,FV3_ML \
    -D32BIT=ON \
    -DMULTI_GASES=OFF

# Build
make -j16
```

## Integration Points

### CCPP Integration

CCPP provides the standard way to add physics. Create CCPP-compliant ML schemes:

```
ccpp-physics/physics/
├── ml_convection.F90         # ML convection scheme
├── ml_convection.meta        # CCPP metadata
├── ml_pbl.F90               # ML PBL scheme
├── ml_pbl.meta              # CCPP metadata
└── ml_radiation_correction.F90  # ML radiation correction
```

### GFS Physics Hierarchy

```
GFS_typedefs.F90             # State and diagnostic types
GFS_diagnostics.F90          # Diagnostic output
GFS_physics_driver.F90       # Physics driver
GFS_suite_*.xml             # Physics suites
```

## Example: CCPP-Compliant ML Physics

### 1. Create ML Convection Scheme

**File: `ccpp-physics/physics/ml_convection.F90`**:

```fortran
!> \file ml_convection.F90
!! This file contains the CCPP-compliant ML convection scheme

module ml_convection

    use machine, only: kind_phys
    use ftorch
    use iso_fortran_env, only: real32

    implicit none

    private
    public :: ml_convection_init, ml_convection_run, ml_convection_finalize

    ! Module-level ML model
    type(torch_model), save :: ml_conv_model
    logical, save :: model_initialized = .false.

contains

!> \section arg_table_ml_convection_init Argument Table
!! \htmlinclude ml_convection_init.html
!!
subroutine ml_convection_init(ml_model_path, mpirank, mpiroot, &
                               errmsg, errflg)

    implicit none

    ! Interface variables
    character(len=*), intent(in) :: ml_model_path
    integer, intent(in) :: mpirank
    integer, intent(in) :: mpiroot
    character(len=*), intent(out) :: errmsg
    integer, intent(out) :: errflg

    ! Initialize error handling
    errmsg = ''
    errflg = 0

    ! Load model (each MPI rank loads its own copy)
    if (mpirank == mpiroot) then
        write(*,'(a,a)') 'Loading ML convection model: ', trim(ml_model_path)
    end if

    ml_conv_model = torch_load_model(trim(ml_model_path), TORCH_DEVICE_CPU)

    if (.not. c_associated(ml_conv_model%ptr)) then
        errmsg = 'Failed to load ML convection model'
        errflg = 1
        return
    end if

    model_initialized = .true.

    if (mpirank == mpiroot) then
        write(*,*) 'ML convection model initialized successfully'
    end if

end subroutine ml_convection_init

!> \section arg_table_ml_convection_run Argument Table
!! \htmlinclude ml_convection_run.html
!!
subroutine ml_convection_run(                                       &
    ! Dimensions
    im, levs, ntrac, nn,                                            &
    ! State variables
    t, q, u, v, prsl, prsi, phil, phii,                            &
    ! Surface fields
    ps, hpbl, slmsk,                                                &
    ! Time step
    dtp,                                                            &
    ! Output tendencies
    dt_t, dq_v, du, dv,                                            &
    ! Output diagnostics
    cnvqc, cnvw, cnvc,                                             &
    ! Error handling
    errmsg, errflg)

    implicit none

    ! Dimensions
    integer, intent(in) :: im      ! Horizontal dimension
    integer, intent(in) :: levs    ! Vertical levels
    integer, intent(in) :: ntrac   ! Number of tracers
    integer, intent(in) :: nn      ! Time level

    ! State variables (im, levs)
    real(kind=kind_phys), intent(in), dimension(im,levs) :: t      ! Temperature
    real(kind=kind_phys), intent(in), dimension(im,levs) :: q      ! Specific humidity
    real(kind=kind_phys), intent(in), dimension(im,levs) :: u, v   ! Winds
    real(kind=kind_phys), intent(in), dimension(im,levs) :: prsl   ! Pressure
    real(kind=kind_phys), intent(in), dimension(im,levs+1) :: prsi ! Interface pressure
    real(kind=kind_phys), intent(in), dimension(im,levs) :: phil   ! Geopotential
    real(kind=kind_phys), intent(in), dimension(im,levs+1) :: phii ! Interface geopotential

    ! Surface fields
    real(kind=kind_phys), intent(in), dimension(im) :: ps          ! Surface pressure
    real(kind=kind_phys), intent(in), dimension(im) :: hpbl        ! PBL height
    real(kind=kind_phys), intent(in), dimension(im) :: slmsk       ! Land-sea mask

    ! Time step
    real(kind=kind_phys), intent(in) :: dtp

    ! Output tendencies
    real(kind=kind_phys), intent(out), dimension(im,levs) :: dt_t  ! T tendency
    real(kind=kind_phys), intent(out), dimension(im,levs) :: dq_v  ! q tendency
    real(kind=kind_phys), intent(out), dimension(im,levs) :: du    ! u tendency
    real(kind=kind_phys), intent(out), dimension(im,levs) :: dv    ! v tendency

    ! Output diagnostics
    real(kind=kind_phys), intent(out), dimension(im,levs) :: cnvqc ! Convective cloud water
    real(kind=kind_phys), intent(out), dimension(im) :: cnvw       ! Convective vertical velocity
    real(kind=kind_phys), intent(out), dimension(im) :: cnvc       ! Convective cloud cover

    ! Error handling
    character(len=*), intent(out) :: errmsg
    integer, intent(out) :: errflg

    ! Local variables
    integer :: i, k
    real(real32), allocatable :: ml_input(:), ml_output(:)
    type(torch_tensor) :: input_tensor, output_tensor
    integer, parameter :: n_input = 64
    integer, parameter :: n_output = 32

    ! Initialize
    errmsg = ''
    errflg = 0

    if (.not. model_initialized) then
        errmsg = 'ML convection model not initialized'
        errflg = 1
        return
    end if

    ! Initialize tendencies
    dt_t = 0.0
    dq_v = 0.0
    du = 0.0
    dv = 0.0

    ! Allocate work arrays
    allocate(ml_input(n_input))
    allocate(ml_output(n_output))

    ! Loop over columns (horizontal points)
    do i = 1, im

        ! Extract column state for ML
        call pack_column_state(t(i,:), q(i,:), u(i,:), v(i,:), &
                               prsl(i,:), ps(i), hpbl(i), slmsk(i), &
                               levs, ml_input)

        ! Run ML inference
        input_tensor = torch_tensor_from_array(ml_input)
        output_tensor = torch_forward(ml_conv_model, input_tensor)
        call torch_tensor_to_array(output_tensor, ml_output)

        ! Unpack ML tendencies
        call unpack_tendencies(ml_output, dt_t(i,:), dq_v(i,:), &
                               levs, dtp)

        ! Cleanup
        call torch_free_tensor(input_tensor)
        call torch_free_tensor(output_tensor)

    end do

    ! Cleanup
    deallocate(ml_input, ml_output)

end subroutine ml_convection_run

!> \section arg_table_ml_convection_finalize Argument Table
!! \htmlinclude ml_convection_finalize.html
!!
subroutine ml_convection_finalize()

    implicit none

    if (model_initialized) then
        call torch_free_model(ml_conv_model)
        model_initialized = .false.
    end if

end subroutine ml_convection_finalize

!-------------------------------------------------------------------
! Helper subroutines
!-------------------------------------------------------------------

subroutine pack_column_state(t, q, u, v, p, ps, pbl_h, lsmask, &
                              nz, state_vec)

    integer, intent(in) :: nz
    real(kind=kind_phys), dimension(nz), intent(in) :: t, q, u, v, p
    real(kind=kind_phys), intent(in) :: ps, pbl_h, lsmask
    real(real32), intent(out) :: state_vec(:)

    integer :: k, idx

    idx = 1

    ! Pack vertical profiles
    do k = 1, nz
        state_vec(idx) = real(t(k), real32)
        idx = idx + 1
    end do

    do k = 1, nz
        state_vec(idx) = real(q(k), real32)
        idx = idx + 1
    end do

    ! Add surface and PBL information
    state_vec(idx) = real(ps, real32)
    state_vec(idx+1) = real(pbl_h, real32)
    state_vec(idx+2) = real(lsmask, real32)

end subroutine pack_column_state

subroutine unpack_tendencies(ml_out, dt_t, dq_v, nz, dt)

    integer, intent(in) :: nz
    real(real32), intent(in) :: ml_out(:)
    real(kind=kind_phys), dimension(nz), intent(out) :: dt_t, dq_v
    real(kind=kind_phys), intent(in) :: dt

    integer :: k, idx

    idx = 1

    ! Unpack temperature tendency
    do k = 1, nz
        dt_t(k) = real(ml_out(idx), kind_phys) / dt
        idx = idx + 1
    end do

    ! Unpack moisture tendency
    do k = 1, nz
        dq_v(k) = real(ml_out(idx), kind_phys) / dt
        idx = idx + 1
    end do

end subroutine unpack_tendencies

end module ml_convection
```

### 2. Create CCPP Metadata

**File: `ccpp-physics/physics/ml_convection.meta`**:

```
[ccpp-table-properties]
  name = ml_convection
  type = scheme
  dependencies = ftorch

[ccpp-arg-table]
  name = ml_convection_init
  type = scheme
[ml_model_path]
  standard_name = ml_convection_model_path
  long_name = path to ML convection model file
  units = none
  dimensions = ()
  type = character
  kind = len=*
  intent = in
  optional = F
[mpirank]
  standard_name = mpi_rank
  long_name = current MPI rank
  units = index
  dimensions = ()
  type = integer
  intent = in
  optional = F
[errmsg]
  standard_name = ccpp_error_message
  long_name = error message for error handling in CCPP
  units = none
  dimensions = ()
  type = character
  kind = len=*
  intent = out
  optional = F
[errflg]
  standard_name = ccpp_error_flag
  long_name = error flag for error handling in CCPP
  units = flag
  dimensions = ()
  type = integer
  intent = out
  optional = F

[ccpp-arg-table]
  name = ml_convection_run
  type = scheme
[im]
  standard_name = horizontal_loop_extent
  long_name = horizontal loop extent
  units = count
  dimensions = ()
  type = integer
  intent = in
  optional = F
[levs]
  standard_name = vertical_dimension
  long_name = vertical layer dimension
  units = count
  dimensions = ()
  type = integer
  intent = in
  optional = F
[t]
  standard_name = air_temperature
  long_name = layer mean air temperature
  units = K
  dimensions = (horizontal_loop_extent,vertical_dimension)
  type = real
  kind = kind_phys
  intent = in
  optional = F
[dt_t]
  standard_name = tendency_of_air_temperature_due_to_ml_convection
  long_name = temperature tendency from ML convection
  units = K s-1
  dimensions = (horizontal_loop_extent,vertical_dimension)
  type = real
  kind = kind_phys
  intent = out
  optional = F
# ... (continue for all variables)
```

### 3. Create Physics Suite

**File: `ccpp-physics/suites/suite_FV3_ML.xml`**:

```xml
<?xml version="1.0" encoding="UTF-8"?>

<suite name="FV3_ML" version="1">
  <!-- Physics suite with ML schemes -->

  <group name="time_vary">
    <subcycle loop="1">
      <scheme>GFS_time_vary_pre</scheme>
      <scheme>GFS_rrtmg_setup</scheme>
    </subcycle>
  </group>

  <group name="radiation">
    <subcycle loop="1">
      <scheme>GFS_suite_interstitial_rad_reset</scheme>
      <scheme>sgscloud_radpre</scheme>
      <scheme>GFS_rrtmg_pre</scheme>
      <scheme>rrtmg_sw</scheme>
      <scheme>rrtmg_lw</scheme>
      <scheme>GFS_rrtmg_post</scheme>
    </subcycle>
  </group>

  <group name="physics">
    <subcycle loop="1">
      <scheme>GFS_suite_interstitial_phys_reset</scheme>
      <scheme>GFS_suite_stateout_reset</scheme>
      <scheme>get_prs_fv3</scheme>
      <scheme>GFS_suite_interstitial_1</scheme>
      <scheme>GFS_surface_generic_pre</scheme>
      <scheme>GFS_surface_composites_pre</scheme>
      <scheme>dcyc2t3</scheme>
      <scheme>GFS_surface_composites_inter</scheme>
      <scheme>GFS_suite_interstitial_2</scheme>
    </subcycle>

    <!-- Deep convection with ML -->
    <subcycle loop="1">
      <scheme>ml_convection</scheme>   <!-- ML scheme! -->
      <scheme>GFS_suite_stateout_update</scheme>
    </subcycle>

    <!-- Continue with other physics -->
    <subcycle loop="1">
      <scheme>samfdeepcnv</scheme>
      <scheme>GFS_suite_interstitial_3</scheme>
      <scheme>samfshalcnv</scheme>
      <scheme>cnvc90</scheme>
      <scheme>GFS_suite_interstitial_4</scheme>
    </subcycle>
  </group>
</suite>
```

### 4. Update Namelist

**File: `input.nml`**:

```fortran
&gfs_physics_nml
  ml_convection_model_path = '/path/to/models/convection_gfs.pt'
  do_ml_convection = .true.
/
```

## Example: ML Post-Processing

ML can enhance model output:

```fortran
module ml_postprocessing

    use ftorch
    use machine, only: kind_phys

    implicit none
    type(torch_model) :: postproc_model

contains

subroutine ml_enhance_output(t2m, rh2m, precip, enhanced_fields)

    ! Take raw model output and enhance using ML
    ! Useful for bias correction, downscaling, derived variables

    real(kind=kind_phys), intent(in) :: t2m(:,:)      ! 2m temperature
    real(kind=kind_phys), intent(in) :: rh2m(:,:)     ! 2m RH
    real(kind=kind_phys), intent(in) :: precip(:,:)   ! Precipitation
    real(kind=kind_phys), intent(out) :: enhanced_fields(:,:,:)

    ! Pack fields, run ML, unpack enhancements
    ! Could include:
    ! - Bias-corrected temperature
    ! - Probability of precipitation
    ! - Gust potential
    ! - Severe weather indices

end subroutine ml_enhance_output

end module ml_postprocessing
```

## Example: Data Assimilation Enhancement

Use ML for observation operators:

```fortran
module ml_observation_operator

    use ftorch

    implicit none
    type(torch_model) :: obs_op_model

contains

subroutine ml_forward_operator(state, obs_type, predicted_obs)

    ! Given model state, predict what observations should be
    ! Useful for non-traditional observations (radar, satellite)

    real(kind=kind_phys), intent(in) :: state(:,:,:)
    integer, intent(in) :: obs_type
    real(kind=kind_phys), intent(out) :: predicted_obs(:)

    ! ML-based observation operator
    ! More accurate than traditional operators for complex obs

end subroutine ml_forward_operator

end module ml_observation_operator
```

## Cubed-Sphere Grid Considerations

### Understanding the Cubed-Sphere

FV3 uses a cubed-sphere grid:
- 6 tiles (faces of a cube)
- Each tile has nx × ny points
- More uniform than lat-lon

### Handling Tile Boundaries

```fortran
subroutine ml_scheme_on_cubed_sphere(state, tile_id, nx, ny)

    integer, intent(in) :: tile_id  ! 1-6
    integer, intent(in) :: nx, ny
    real(kind=kind_phys), intent(inout) :: state(:,:,:)

    ! Process each tile
    ! Be aware of tile boundaries for schemes needing neighbors

    do j = 1, ny
        do i = 1, nx
            ! Extract column
            ! Run ML
            ! Apply result
        end do
    end do

end subroutine ml_scheme_on_cubed_sphere
```

### Tile-Aware Training

Train ML models with tile information:

```python
# Include tile ID and position as features
class CubedSphereAwareModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input includes tile_id, i_pos, j_pos
        self.embedding = nn.Embedding(6, 8)  # Tile embeddings
        self.fc = nn.Linear(state_size + 8 + 2, output_size)

    def forward(self, state, tile_id, i_pos, j_pos):
        tile_embed = self.embedding(tile_id)
        pos = torch.stack([i_pos, j_pos], dim=1)
        x = torch.cat([state, tile_embed, pos], dim=1)
        return self.fc(x)
```

## Operational Deployment

### 1. Model Management

```bash
# Operational model directory structure
/ops/models/ml/
├── convection/
│   ├── v1.0/
│   │   └── model.pt
│   └── v1.1/
│       └── model.pt
├── pbl/
└── radiation/
```

### 2. Version Control

```fortran
module ml_model_version

contains

subroutine check_ml_model_version(model_path, required_version)

    character(len=*), intent(in) :: model_path
    character(len=*), intent(in) :: required_version

    ! Verify model version matches expectations
    ! Critical for operational consistency

end subroutine check_ml_model_version

end module ml_model_version
```

### 3. Fail-Safe Mechanisms

```fortran
subroutine ml_convection_with_fallback(...)

    integer :: ml_status

    ! Try ML scheme
    call ml_convection_run(..., errflg=ml_status)

    if (ml_status /= 0) then
        ! Fall back to traditional scheme
        write(*,*) 'WARNING: ML convection failed, using fallback'
        call samfdeepcnv(...)  ! Traditional scheme
    end if

end subroutine ml_convection_with_fallback
```

### 4. Performance Monitoring

```fortran
module ml_performance_stats

    real(kind=kind_phys), save :: ml_total_time = 0.0
    integer, save :: ml_call_count = 0

contains

subroutine ml_performance_report()

    real(kind=kind_phys) :: avg_time

    if (ml_call_count > 0) then
        avg_time = ml_total_time / real(ml_call_count)
        write(*,'(A,F10.4,A)') 'ML avg time per call: ', avg_time*1000, ' ms'
        write(*,'(A,I0)') 'Total ML calls: ', ml_call_count
    end if

end subroutine ml_performance_report

end module ml_performance_stats
```

## Testing and Validation

### 1. C96 Test Case

```bash
# Standard FV3 resolution
cd fv3gfs/tests
./compile.sh
./rt.sh -c -l fv3_ccpp_gfs_ml  # Test with ML physics
```

### 2. Compare with Baseline

```bash
# Run baseline
./fv3.exe < input.nml.baseline

# Run with ML
./fv3.exe < input.nml.ml

# Compare outputs
python compare_forecast.py GFSPRS.GrbF00 GFSPRS_ML.GrbF00
```

### 3. Operational Verification

```python
import pygrib
import numpy as np
from sklearn.metrics import mean_squared_error

# Load forecasts
grbs_baseline = pygrib.open('gfs.t00z.pgrb2.0p25.f024')
grbs_ml = pygrib.open('gfs_ml.t00z.pgrb2.0p25.f024')

# Load analysis for verification
grbs_analysis = pygrib.open('gdas.t00z.pgrb2.0p25.f000')

# Compute scores
def compute_rmse(forecast, analysis, variable):
    fcst = forecast.select(name=variable)[0].values
    anal = analysis.select(name=variable)[0].values
    return np.sqrt(mean_squared_error(fcst, anal))

rmse_baseline = compute_rmse(grbs_baseline, grbs_analysis, '2 metre temperature')
rmse_ml = compute_rmse(grbs_ml, grbs_analysis, '2 metre temperature')

print(f'Baseline RMSE: {rmse_baseline:.2f} K')
print(f'ML RMSE: {rmse_ml:.2f} K')
print(f'Improvement: {(rmse_baseline - rmse_ml)/rmse_baseline * 100:.1f}%')
```

## HPC Deployment

### NOAA RDHPCS

```bash
# On Hera/Jet
module load intel/2021.3.0
module load impi/2021.3.0

# Set library paths
export LD_LIBRARY_PATH=/apps/fortran-torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/apps/libtorch/lib:$LD_LIBRARY_PATH

# Run
srun -n 240 ./fv3.exe
```

### Memory Management

```fortran
! Monitor memory usage in operational setting
call getrusage(RUSAGE_SELF, usage)
max_rss = usage%ru_maxrss

if (max_rss > memory_limit) then
    write(*,*) 'WARNING: High memory usage from ML:', max_rss
end if
```

## Best Practices for FV3

1. **CCPP Compliance**: Follow CCPP standards for interoperability
2. **Error Handling**: Use CCPP error framework
3. **Thread Safety**: Handle OpenMP and MPI correctly
4. **Fail-Safe**: Always have fallback to traditional physics
5. **Monitoring**: Log ML performance and errors
6. **Versioning**: Track ML model versions
7. **Testing**: Use regression test framework

## Troubleshooting

**Issue: CCPP initialization error**

```fortran
! Check metadata matches code
! Verify all required variables are provided
! Ensure proper intent declarations
```

**Issue: MPI deadlock**

```fortran
! Each rank must load its own model
! Avoid MPI communication during ML inference
call MPI_Barrier(MPI_COMM_WORLD, ierr)
```

**Issue: Performance degradation**

```fortran
! Profile ML overhead
! Consider batching
! Check thread scaling
```

## Summary

FV3 Integration Checklist:

1. ✓ Build Fortran-Torch compatible with FV3 compilers
2. ✓ Create CCPP-compliant ML scheme with metadata
3. ✓ Register in physics suite XML
4. ✓ Handle cubed-sphere grid properly
5. ✓ Implement fail-safe mechanisms
6. ✓ Test with FV3 regression tests
7. ✓ Deploy with monitoring

Key advantage: **Operational-ready** ML integration with CCPP framework compatibility!

## References

- FV3 Documentation: https://noaa-emc.github.io/FV3_Dycore_ufs-v2.0.0/html/
- CCPP Documentation: https://ccpp-techdoc.readthedocs.io/
- UFS Weather Model: https://ufs-weather-model.readthedocs.io/
- Fortran-Torch: https://github.com/fzhao70/Fortran-Torch

---

For operational support, contact NOAA/EMC or see [GitHub Issues](https://github.com/fzhao70/Fortran-Torch/issues).

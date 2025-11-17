# Integrating Fortran-Torch into WRF (Weather Research and Forecasting Model)

This guide shows how to integrate PyTorch machine learning models into the WRF model using Fortran-Torch.

## Table of Contents

- [Overview](#overview)
- [Build Configuration](#build-configuration)
- [Integration Points](#integration-points)
- [Example: ML-based Convection Scheme](#example-ml-based-convection-scheme)
- [Example: PBL Parameterization](#example-pbl-parameterization)
- [Example: Microphysics Enhancement](#example-microphysics-enhancement)
- [Performance Considerations](#performance-considerations)
- [Testing and Validation](#testing-and-validation)

## Overview

WRF is a mesoscale numerical weather prediction system designed for both atmospheric research and operational forecasting. Fortran-Torch enables integration of ML models for:

- **Physics Parameterizations**: Replace or augment schemes (convection, PBL, microphysics)
- **Bias Correction**: Correct systematic model biases
- **Subgrid Processes**: Parameterize unresolved processes
- **Post-processing**: Enhance model output

### WRF Repository
- **GitHub**: https://github.com/wrf-model/WRF
- **Documentation**: https://www2.mmm.ucar.edu/wrf/users/
- **Version**: Compatible with WRF v4.x+

## Build Configuration

### 1. Build Fortran-Torch

First, build Fortran-Torch with the same compiler you'll use for WRF:

```bash
# Download LibTorch
cd /path/to/Fortran-Torch
./scripts/download_libtorch.sh cpu  # or cu118 for GPU

# Build with WRF-compatible compiler
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_Fortran_COMPILER=gfortran \  # Match WRF compiler
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_INSTALL_PREFIX=/opt/fortran-torch \
      ..
make -j8
sudo make install
```

### 2. Modify WRF Build System

Add Fortran-Torch to WRF's configure file:

**Edit `configure.wrf` (after running `./configure`)**:

```makefile
# Add Fortran-Torch paths
FTORCH_DIR = /opt/fortran-torch
LIBTORCH_DIR = /path/to/libtorch

# Add to include paths
INCLUDE_MODULES = ... -I$(FTORCH_DIR)/include/fortran

# Add to library paths and libraries
LIB_EXTERNAL = ... -L$(FTORCH_DIR)/lib -lftorch -lfortran_torch_cpp \
               -L$(LIBTORCH_DIR)/lib -ltorch -ltorch_cpu -lc10 \
               -Wl,-rpath,$(LIBTORCH_DIR)/lib -Wl,-rpath,$(FTORCH_DIR)/lib
```

**Alternative: Modify `arch/configure.defaults`** (before ./configure):

```bash
# Find your architecture section and add:
FTORCH_LIBS = -L/opt/fortran-torch/lib -lftorch -lfortran_torch_cpp \
              -L/path/to/libtorch/lib -ltorch -ltorch_cpu -lc10

# Add to LDFLAGS
LDFLAGS_LOCAL = ... $(FTORCH_LIBS)
```

### 3. Verify Build

```bash
cd WRF
./clean -a
./configure  # Select your platform
# Edit configure.wrf as above
./compile em_real >& compile.log
```

## Integration Points

### WRF Directory Structure

```
WRF/
├── phys/              # Physics schemes (main integration point)
│   ├── module_cu_*.F  # Cumulus schemes
│   ├── module_bl_*.F  # Boundary layer schemes
│   ├── module_mp_*.F  # Microphysics schemes
│   └── module_ra_*.F  # Radiation schemes
├── dyn_em/            # Dynamics
└── Registry/          # Variable registry
```

### Recommended Integration Points

1. **Physics Schemes** (`phys/`)
   - Add ML parameterizations alongside existing schemes
   - Create new module: `module_cu_ml.F` for ML convection

2. **Registry** (`Registry/Registry.EM_COMMON`)
   - Define new variables for ML model state
   - Add namelist options for ML schemes

3. **Initialization** (`dyn_em/start_em.F`)
   - Load ML models during initialization

## Example: ML-based Convection Scheme

### 1. Create ML Convection Module

Create `phys/module_cu_ml.F`:

```fortran
!-------------------------------------------------------------------
! MODULE: module_cu_ml
!
! DESCRIPTION:
!   Machine learning-based cumulus parameterization using Fortran-Torch
!
! AUTHOR: Fanghe Zhao
! DATE: 2024
!-------------------------------------------------------------------

MODULE module_cu_ml

    USE module_model_constants
    USE ftorch  ! Fortran-Torch module
    USE iso_fortran_env, only: real32

    IMPLICIT NONE

    PRIVATE
    PUBLIC :: cu_ml_driver, cu_ml_init, cu_ml_finalize

    ! Module-level variables
    TYPE(torch_model) :: convection_model
    LOGICAL :: model_loaded = .FALSE.
    INTEGER :: model_input_size
    INTEGER :: model_output_size

CONTAINS

!-------------------------------------------------------------------
! SUBROUTINE: cu_ml_init
!
! Initialize ML convection scheme
!-------------------------------------------------------------------
SUBROUTINE cu_ml_init(model_path, cuda_flag)

    IMPLICIT NONE

    CHARACTER(LEN=*), INTENT(IN) :: model_path
    LOGICAL, INTENT(IN), OPTIONAL :: cuda_flag
    INTEGER(torch_device) :: device

    ! Determine device
    IF (PRESENT(cuda_flag) .AND. cuda_flag) THEN
        device = TORCH_DEVICE_CUDA
        WRITE(*,*) 'ML Convection: Using CUDA'
    ELSE
        device = TORCH_DEVICE_CPU
        WRITE(*,*) 'ML Convection: Using CPU'
    END IF

    ! Load model
    WRITE(*,*) 'Loading ML convection model from: ', TRIM(model_path)
    convection_model = torch_load_model(TRIM(model_path), device)

    IF (c_associated(convection_model%ptr)) THEN
        model_loaded = .TRUE.
        WRITE(*,*) 'ML convection model loaded successfully'
    ELSE
        WRITE(*,*) 'ERROR: Failed to load ML convection model'
        STOP 1
    END IF

    ! Set model dimensions (example)
    model_input_size = 50   ! Temperature, humidity, winds at levels
    model_output_size = 20  ! Heating rates, moisture tendencies

END SUBROUTINE cu_ml_init

!-------------------------------------------------------------------
! SUBROUTINE: cu_ml_driver
!
! Main driver for ML convection scheme
!-------------------------------------------------------------------
SUBROUTINE cu_ml_driver(                                     &
              ! Input fields
              t3d, qv3d, qc3d, qi3d, p3d, pi3d, rho,        &
              ! Output tendencies
              rthcuten, rqvcuten, rqccuten,                  &
              ! Dimensions
              ids, ide, jds, jde, kds, kde,                  &
              ims, ime, jms, jme, kms, kme,                  &
              its, ite, jts, jte, kts, kte,                  &
              ! Time step
              dt )

    IMPLICIT NONE

    ! Dimensions
    INTEGER, INTENT(IN) :: ids, ide, jds, jde, kds, kde
    INTEGER, INTENT(IN) :: ims, ime, jms, jme, kms, kme
    INTEGER, INTENT(IN) :: its, ite, jts, jte, kts, kte

    ! Input fields
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(IN) :: t3d
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(IN) :: qv3d
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(IN) :: qc3d
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(IN) :: qi3d
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(IN) :: p3d
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(IN) :: pi3d
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(IN) :: rho

    ! Output tendencies
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(OUT) :: rthcuten
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(OUT) :: rqvcuten
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(OUT) :: rqccuten

    ! Time step
    REAL, INTENT(IN) :: dt

    ! Local variables
    INTEGER :: i, j, k
    REAL(real32), ALLOCATABLE :: ml_input(:), ml_output(:)
    TYPE(torch_tensor) :: input_tensor, output_tensor
    INTEGER :: nz

    ! Check if model is loaded
    IF (.NOT. model_loaded) THEN
        WRITE(*,*) 'ERROR: ML model not loaded'
        RETURN
    END IF

    ! Allocate work arrays
    ALLOCATE(ml_input(model_input_size))
    ALLOCATE(ml_output(model_output_size))

    ! Initialize tendencies
    rthcuten = 0.0
    rqvcuten = 0.0
    rqccuten = 0.0

    ! Loop over horizontal grid points
    DO j = jts, jte
        DO i = its, ite

            ! Extract column state for ML model
            CALL extract_column_state(t3d(i,:,j), qv3d(i,:,j), qc3d(i,:,j), &
                                      qi3d(i,:,j), p3d(i,:,j), pi3d(i,:,j),  &
                                      kts, kte, ml_input)

            ! Create input tensor
            input_tensor = torch_tensor_from_array(ml_input)

            ! Run ML inference
            output_tensor = torch_forward(convection_model, input_tensor)

            ! Extract output
            CALL torch_tensor_to_array(output_tensor, ml_output)

            ! Apply ML tendencies to column
            CALL apply_ml_tendencies(ml_output, rthcuten(i,:,j), &
                                     rqvcuten(i,:,j), rqccuten(i,:,j), &
                                     kts, kte, dt)

            ! Free tensors
            CALL torch_free_tensor(input_tensor)
            CALL torch_free_tensor(output_tensor)

        END DO
    END DO

    ! Cleanup
    DEALLOCATE(ml_input, ml_output)

END SUBROUTINE cu_ml_driver

!-------------------------------------------------------------------
! SUBROUTINE: extract_column_state
!
! Extract atmospheric column state for ML model input
!-------------------------------------------------------------------
SUBROUTINE extract_column_state(t, qv, qc, qi, p, pi, kts, kte, state_vector)

    IMPLICIT NONE

    INTEGER, INTENT(IN) :: kts, kte
    REAL, DIMENSION(kts:kte), INTENT(IN) :: t, qv, qc, qi, p, pi
    REAL(real32), INTENT(OUT) :: state_vector(:)

    INTEGER :: k, idx

    ! Pack state variables
    idx = 1

    ! Temperature
    DO k = kts, kte
        state_vector(idx) = REAL(t(k), real32)
        idx = idx + 1
    END DO

    ! Water vapor mixing ratio
    DO k = kts, kte
        state_vector(idx) = REAL(qv(k), real32)
        idx = idx + 1
    END DO

    ! Cloud water
    DO k = kts, kte
        state_vector(idx) = REAL(qc(k), real32)
        idx = idx + 1
    END DO

    ! Add more variables as needed...

END SUBROUTINE extract_column_state

!-------------------------------------------------------------------
! SUBROUTINE: apply_ml_tendencies
!
! Apply ML-predicted tendencies to WRF state
!-------------------------------------------------------------------
SUBROUTINE apply_ml_tendencies(ml_output, rthcuten, rqvcuten, rqccuten, &
                                kts, kte, dt)

    IMPLICIT NONE

    INTEGER, INTENT(IN) :: kts, kte
    REAL(real32), INTENT(IN) :: ml_output(:)
    REAL, DIMENSION(kts:kte), INTENT(OUT) :: rthcuten, rqvcuten, rqccuten
    REAL, INTENT(IN) :: dt

    INTEGER :: k, idx
    REAL :: scale_factor

    ! Scaling factor for tendencies (model-dependent)
    scale_factor = 1.0 / dt

    idx = 1

    ! Temperature tendencies (K/s)
    DO k = kts, kte
        rthcuten(k) = REAL(ml_output(idx)) * scale_factor
        idx = idx + 1
    END DO

    ! Moisture tendencies (kg/kg/s)
    DO k = kts, kte
        rqvcuten(k) = REAL(ml_output(idx)) * scale_factor
        idx = idx + 1
    END DO

    ! Additional tendencies...

END SUBROUTINE apply_ml_tendencies

!-------------------------------------------------------------------
! SUBROUTINE: cu_ml_finalize
!
! Clean up ML convection scheme
!-------------------------------------------------------------------
SUBROUTINE cu_ml_finalize()

    IMPLICIT NONE

    IF (model_loaded) THEN
        CALL torch_free_model(convection_model)
        model_loaded = .FALSE.
        WRITE(*,*) 'ML convection model freed'
    END IF

END SUBROUTINE cu_ml_finalize

END MODULE module_cu_ml
```

### 2. Register the New Scheme

**Edit `Registry/Registry.EM_COMMON`**:

Add namelist option:

```fortran
# ML Convection Scheme
rconfig integer cu_physics_ml namelist,physics 0 0
rconfig character ml_cu_model_path namelist,physics max_domains "convection_model.pt"
```

### 3. Integrate into Physics Driver

**Edit `phys/module_physics_init.F`**:

```fortran
USE module_cu_ml  ! Add this

! In physics_init subroutine:
IF (config_flags%cu_physics_ml > 0) THEN
    CALL cu_ml_init(TRIM(config_flags%ml_cu_model_path), &
                    cuda_flag=.FALSE.)
END IF
```

**Edit `dyn_em/solve_em.F`**:

```fortran
! In the physics section, add:
IF (config_flags%cu_physics == CU_ML) THEN
    CALL cu_ml_driver(                              &
         t_phy, qv_curr, qc_curr, qi_curr,         &
         p_phy, pi_phy, rho,                        &
         rthcuten, rqvcuten, rqccuten,              &
         ids, ide, jds, jde, kds, kde,              &
         ims, ime, jms, jme, kms, kme,              &
         its, ite, jts, jte, kts, kte,              &
         dt )
END IF
```

### 4. Update namelist.input

```fortran
&physics
 cu_physics_ml                = 1,    ! Enable ML convection
 ml_cu_model_path             = '/path/to/convection_model.pt',
/
```

## Example: PBL Parameterization

Create `phys/module_bl_ml.F` for ML-based planetary boundary layer scheme:

```fortran
MODULE module_bl_ml

    USE ftorch
    USE module_model_constants

    IMPLICIT NONE
    PRIVATE
    PUBLIC :: bl_ml_driver, bl_ml_init

    TYPE(torch_model) :: pbl_model
    LOGICAL :: pbl_model_loaded = .FALSE.

CONTAINS

SUBROUTINE bl_ml_init(model_path)
    CHARACTER(LEN=*), INTENT(IN) :: model_path

    pbl_model = torch_load_model(TRIM(model_path), TORCH_DEVICE_CPU)

    IF (c_associated(pbl_model%ptr)) THEN
        pbl_model_loaded = .TRUE.
        WRITE(*,*) 'ML PBL model loaded'
    END IF
END SUBROUTINE bl_ml_init

SUBROUTINE bl_ml_driver(                                    &
              ! Input
              u3d, v3d, th3d, qv3d, p3d, pi3d,             &
              ! Surface fields
              ust, tsk, qsfc, ps, hfx, qfx,                &
              ! Output tendencies
              rublten, rvblten, rthblten, rqvblten,        &
              ! Dimensions and control
              ids, ide, jds, jde, kds, kde,                &
              ims, ime, jms, jme, kms, kme,                &
              its, ite, jts, jte, kts, kte,                &
              dt )

    ! Similar structure to convection scheme
    ! Extract PBL column state (winds, temperature, moisture, surface fluxes)
    ! Run ML model
    ! Apply mixing/tendency corrections

END SUBROUTINE bl_ml_driver

END MODULE module_bl_ml
```

## Example: Microphysics Enhancement

Enhance existing microphysics with ML correction:

```fortran
MODULE module_mp_ml_enhancement

    USE ftorch
    USE module_mp_thompson  ! Base Thompson scheme

    IMPLICIT NONE
    TYPE(torch_model) :: mp_correction_model

CONTAINS

SUBROUTINE mp_ml_enhanced_driver(...)

    ! Call base Thompson scheme
    CALL mp_gt_driver(...)

    ! Apply ML correction
    CALL apply_ml_mp_correction(qv, qc, qr, qi, qs, qg, ...)

END SUBROUTINE mp_ml_enhanced_driver

SUBROUTINE apply_ml_mp_correction(qv, qc, qr, qi, qs, qg, ...)

    ! Extract hydrometeor state
    ! Run ML model to predict corrections
    ! Apply corrections to mixing ratios

END SUBROUTINE apply_ml_mp_correction

END MODULE module_mp_ml_enhancement
```

## Performance Considerations

### 1. Minimize Data Transfer

```fortran
! Bad: Creating tensors every call
DO timestep = 1, n_timesteps
    DO i = 1, nx
        tensor = torch_tensor_from_array(data(i,:))  ! Slow
    END DO
END DO

! Good: Batch processing
CALL batch_process_columns(data, nx, nz)
```

### 2. Model Placement

```fortran
! For CPU clusters: Load one model per MPI rank
CALL mpi_comm_rank(comm, myrank, ierr)
IF (myrank == 0) WRITE(*,*) 'Loading ML model on rank', myrank
ml_model = torch_load_model('model.pt', TORCH_DEVICE_CPU)
```

### 3. Timing and Profiling

```fortran
REAL(KIND=8) :: ml_time_start, ml_time_end, ml_time_total

CALL cpu_time(ml_time_start)
output_tensor = torch_forward(model, input_tensor)
CALL cpu_time(ml_time_end)

ml_time_total = ml_time_total + (ml_time_end - ml_time_start)

! Print timing statistics periodically
IF (MOD(current_step, 100) == 0) THEN
    WRITE(*,'(A,F10.4,A)') 'ML inference time: ', &
                            ml_time_total/100.0, ' s per step'
    ml_time_total = 0.0
END IF
```

### 4. OpenMP Threading

WRF uses OpenMP. Ensure thread safety:

```fortran
!$OMP PARALLEL DO PRIVATE(i, j, ml_input, ml_output, input_tensor, output_tensor)
DO j = jts, jte
    DO i = its, ite
        ! Each thread needs its own tensors
        CALL extract_column_state(...)
        input_tensor = torch_tensor_from_array(ml_input)
        output_tensor = torch_forward(model, input_tensor)
        CALL torch_tensor_to_array(output_tensor, ml_output)
        CALL torch_free_tensor(input_tensor)
        CALL torch_free_tensor(output_tensor)
    END DO
END DO
!$OMP END PARALLEL DO
```

## Testing and Validation

### 1. Idealized Test Case

Test with WRF's `em_quarter_ss` (supercell) case:

```bash
cd WRF/test/em_quarter_ss
ln -s ../../../DATA/* .
./ideal.exe
./wrf.exe
```

### 2. Verification Script

Create `verify_ml_integration.py`:

```python
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# Read WRF output
ds = nc.Dataset('wrfout_d01_2001-06-11_12:00:00')

# Check ML tendency variables
if 'RTHCUTEN_ML' in ds.variables:
    ml_heating = ds.variables['RTHCUTEN_ML'][:]
    print(f"ML heating range: {ml_heating.min():.2e} to {ml_heating.max():.2e}")

    # Plot vertical profile
    plt.plot(ml_heating[0, :, 50, 50])
    plt.xlabel('Level')
    plt.ylabel('ML Heating Rate (K/s)')
    plt.savefig('ml_heating_profile.png')
```

### 3. Regression Testing

Compare against baseline WRF:

```bash
# Run baseline
./wrf.exe
mv wrfout* wrfout_baseline

# Run with ML
# Edit namelist.input to enable ML scheme
./wrf.exe
mv wrfout* wrfout_ml

# Compare
python compare_runs.py wrfout_baseline wrfout_ml
```

## Troubleshooting

### Common Issues

**1. Segmentation Fault on Model Load**
```bash
# Check library paths
ldd wrf.exe | grep torch
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

**2. MPI Hangs**
```fortran
! Ensure all ranks load the model
CALL mpi_barrier(comm, ierr)
CALL cu_ml_init(...)
CALL mpi_barrier(comm, ierr)
```

**3. Poor Performance**
```fortran
! Profile to find bottleneck
CALL system_clock(count_start, count_rate)
! ... ML code ...
CALL system_clock(count_end)
elapsed = REAL(count_end - count_start) / REAL(count_rate)
```

## References

- WRF User Guide: https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/
- WRF Physics: https://www2.mmm.ucar.edu/wrf/users/physics/phys_refs.html
- Fortran-Torch: https://github.com/fzhao70/Fortran-Torch

## Example Workflow Summary

1. Train ML model in Python (convection, PBL, etc.)
2. Export to TorchScript (`.pt` file)
3. Create WRF physics module using Fortran-Torch
4. Register in WRF Registry
5. Modify configure.wrf for linking
6. Rebuild WRF
7. Update namelist.input
8. Run and validate

---

For questions or issues, see [GitHub Issues](https://github.com/fzhao70/Fortran-Torch/issues).

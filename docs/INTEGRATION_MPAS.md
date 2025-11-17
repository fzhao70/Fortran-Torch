# Integrating Fortran-Torch into MPAS (Model for Prediction Across Scales)

This guide shows how to integrate PyTorch machine learning models into MPAS using Fortran-Torch.

## Table of Contents

- [Overview](#overview)
- [MPAS Architecture](#mpas-architecture)
- [Build Configuration](#build-configuration)
- [Integration Points](#integration-points)
- [Example: ML Convection on Unstructured Mesh](#example-ml-convection-on-unstructured-mesh)
- [Example: Subgrid Turbulence Parameterization](#example-subgrid-turbulence-parameterization)
- [Handling Unstructured Grids](#handling-unstructured-grids)
- [Performance on Unstructured Mesh](#performance-on-unstructured-mesh)
- [Testing and Validation](#testing-and-validation)

## Overview

MPAS is a novel climate and weather model using unstructured Voronoi meshes with variable resolution. Fortran-Torch enables ML integration for:

- **Physics Parameterizations**: Convection, microphysics, PBL on unstructured grids
- **Scale-aware Parameterizations**: ML models that adapt to varying resolution
- **Mesh-specific Corrections**: Bias correction accounting for local resolution
- **Subgrid Processes**: Parameterize turbulence and mixing

### MPAS Repository
- **GitHub**: https://github.com/MPAS-Dev/MPAS-Model
- **Atmosphere Core**: `src/core_atmosphere/`
- **Version**: Compatible with MPAS v7.x+

## MPAS Architecture

MPAS uses a unique structure:

```
MPAS-Model/
├── src/
│   ├── core_atmosphere/       # Atmospheric dynamics and physics
│   │   ├── physics/           # Physics parameterizations
│   │   │   ├── mpas_atmphys_driver.F
│   │   │   ├── mpas_atmphys_camrad.F
│   │   │   └── mpas_atmphys_control.F
│   │   ├── dynamics/          # Atmospheric dynamics
│   │   └── Registry.xml       # Variable registry
│   ├── framework/             # MPAS framework
│   └── operators/             # Grid operators
```

### Key Differences from WRF

1. **Unstructured Mesh**: Voronoi cells instead of regular lat-lon
2. **Variable Resolution**: Resolution varies across domain
3. **Column Physics**: Similar to WRF but on unstructured grid
4. **Registry System**: XML-based variable registration

## Build Configuration

### 1. Build Fortran-Torch

```bash
# Build Fortran-Torch
cd /path/to/Fortran-Torch
./scripts/download_libtorch.sh cpu

mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_Fortran_COMPILER=gfortran \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_INSTALL_PREFIX=/opt/fortran-torch \
      ..
make -j8
sudo make install
```

### 2. Modify MPAS Makefile

**Edit `Makefile` in MPAS-Model root**:

```makefile
# Add Fortran-Torch configuration
FTORCH_DIR = /opt/fortran-torch
LIBTORCH_DIR = /path/to/libtorch

# Add to include paths
override FCINCLUDES += -I$(FTORCH_DIR)/include/fortran

# Add to linker flags
override LIBS += -L$(FTORCH_DIR)/lib -lftorch -lfortran_torch_cpp \
                 -L$(LIBTORCH_DIR)/lib -ltorch -ltorch_cpu -lc10 \
                 -Wl,-rpath,$(LIBTORCH_DIR)/lib

# For C++ compilation
override CXXFLAGS += -I$(FTORCH_DIR)/include
override CXXLIBS += -L$(LIBTORCH_DIR)/lib -ltorch -lc10
```

### 3. Build MPAS

```bash
cd MPAS-Model
make clean CORE=atmosphere
make gfortran CORE=atmosphere USE_PIO2=true
```

## Integration Points

### Physics Integration Points

```
src/core_atmosphere/physics/
├── mpas_atmphys_driver.F           # Main physics driver
├── mpas_atmphys_camrad.F           # Radiation
├── mpas_atmphys_cloudiness.F       # Clouds
├── mpas_atmphys_control.F          # Physics control
├── mpas_atmphys_driver_convection.F # Convection driver
├── mpas_atmphys_driver_pbl.F       # PBL driver
└── mpas_atmphys_vars.F             # Physics variables
```

### Recommended Approach

1. Create new ML physics modules in `physics/`
2. Call from existing drivers (`mpas_atmphys_driver_*.F`)
3. Register variables in `Registry.xml`
4. Add namelist options in `namelist.atmosphere`

## Example: ML Convection on Unstructured Mesh

### 1. Create ML Convection Module

Create `src/core_atmosphere/physics/mpas_atmphys_ml_convection.F`:

```fortran
!==================================================================
! MODULE: mpas_atmphys_ml_convection
!
! DESCRIPTION:
!   ML-based convection scheme for MPAS using Fortran-Torch
!   Handles unstructured mesh with variable resolution
!
! AUTHOR: Fanghe Zhao
! DATE: 2024
!==================================================================

module mpas_atmphys_ml_convection

    use mpas_kind_types
    use mpas_derived_types
    use mpas_pool_routines
    use ftorch  ! Fortran-Torch
    use iso_fortran_env, only: real32

    implicit none
    private
    public :: mpas_atmphys_ml_convection_init, &
              mpas_atmphys_ml_convection_driver, &
              mpas_atmphys_ml_convection_finalize

    ! Module variables
    type(torch_model) :: ml_cu_model
    logical :: ml_cu_initialized = .false.
    integer :: ml_input_size = 50
    integer :: ml_output_size = 20

contains

!------------------------------------------------------------------
! SUBROUTINE: mpas_atmphys_ml_convection_init
!
! Initialize ML convection scheme
!------------------------------------------------------------------
subroutine mpas_atmphys_ml_convection_init(dminfo, configs)

    implicit none

    type(dm_info), intent(in) :: dminfo
    type(mpas_pool_type), intent(inout) :: configs

    character(len=StrKIND) :: model_path
    integer :: ierr

    ! Get model path from namelist
    call mpas_pool_get_config(configs, 'config_ml_cu_model_path', model_path)

    ! Master process loads and broadcasts model info
    if (dminfo % my_proc_id == IO_NODE) then
        write(0,*) '*** Loading ML convection model: ', trim(model_path)
        ml_cu_model = torch_load_model(trim(model_path), TORCH_DEVICE_CPU)

        if (c_associated(ml_cu_model%ptr)) then
            ml_cu_initialized = .true.
            write(0,*) '*** ML convection model loaded successfully'
        else
            write(0,*) '*** ERROR: Failed to load ML convection model'
        end if
    end if

    ! Note: For distributed memory, each process loads its own model
    ! to avoid communication overhead during inference
    if (dminfo % my_proc_id /= IO_NODE) then
        ml_cu_model = torch_load_model(trim(model_path), TORCH_DEVICE_CPU)
        if (c_associated(ml_cu_model%ptr)) ml_cu_initialized = .true.
    end if

end subroutine mpas_atmphys_ml_convection_init

!------------------------------------------------------------------
! SUBROUTINE: mpas_atmphys_ml_convection_driver
!
! Main driver for ML convection on unstructured mesh
!------------------------------------------------------------------
subroutine mpas_atmphys_ml_convection_driver(                  &
                mesh, state, time_lev, diag_physics, tend_physics, &
                configs, dt)

    implicit none

    ! Arguments
    type(mpas_pool_type), intent(in) :: mesh
    type(mpas_pool_type), intent(in) :: state
    integer, intent(in) :: time_lev
    type(mpas_pool_type), intent(inout) :: diag_physics
    type(mpas_pool_type), intent(inout) :: tend_physics
    type(mpas_pool_type), intent(in) :: configs
    real(kind=RKIND), intent(in) :: dt

    ! Local pointers to mesh
    integer, pointer :: nCells, nVertLevels
    integer, dimension(:), pointer :: nEdgesOnCell
    real(kind=RKIND), dimension(:), pointer :: meshDensity

    ! Local pointers to state
    real(kind=RKIND), dimension(:,:), pointer :: theta
    real(kind=RKIND), dimension(:,:), pointer :: qv
    real(kind=RKIND), dimension(:,:), pointer :: pressure
    real(kind=RKIND), dimension(:,:), pointer :: rho

    ! Local pointers to tendencies
    real(kind=RKIND), dimension(:,:), pointer :: tend_theta
    real(kind=RKIND), dimension(:,:), pointer :: tend_qv

    ! Local variables
    integer :: iCell, k
    real(real32), allocatable :: ml_input(:), ml_output(:)
    type(torch_tensor) :: input_tensor, output_tensor
    real(kind=RKIND) :: cell_resolution

    if (.not. ml_cu_initialized) return

    ! Get mesh dimensions
    call mpas_pool_get_dimension(mesh, 'nCells', nCells)
    call mpas_pool_get_dimension(mesh, 'nVertLevels', nVertLevels)

    ! Get mesh information
    call mpas_pool_get_array(mesh, 'meshDensity', meshDensity)

    ! Get state variables
    call mpas_pool_get_array(state, 'theta', theta, time_lev)
    call mpas_pool_get_array(state, 'qv', qv, time_lev)
    call mpas_pool_get_array(state, 'pressure', pressure, time_lev)
    call mpas_pool_get_array(state, 'rho', rho, time_lev)

    ! Get tendency arrays
    call mpas_pool_get_array(tend_physics, 'tend_theta', tend_theta)
    call mpas_pool_get_array(tend_physics, 'tend_qv', tend_qv)

    ! Allocate work arrays
    allocate(ml_input(ml_input_size))
    allocate(ml_output(ml_output_size))

    ! Loop over cells in unstructured mesh
    do iCell = 1, nCells

        ! Get local mesh resolution (important for variable resolution!)
        cell_resolution = 1.0_RKIND / sqrt(meshDensity(iCell))

        ! Extract column state including resolution information
        call extract_column_for_ml(theta(:,iCell), qv(:,iCell), &
                                    pressure(:,iCell), rho(:,iCell), &
                                    cell_resolution, nVertLevels, &
                                    ml_input)

        ! Create tensor and run inference
        input_tensor = torch_tensor_from_array(ml_input)
        output_tensor = torch_forward(ml_cu_model, input_tensor)
        call torch_tensor_to_array(output_tensor, ml_output)

        ! Apply ML tendencies
        call apply_ml_cu_tendencies(ml_output, tend_theta(:,iCell), &
                                     tend_qv(:,iCell), nVertLevels, dt)

        ! Free tensors
        call torch_free_tensor(input_tensor)
        call torch_free_tensor(output_tensor)

    end do

    ! Cleanup
    deallocate(ml_input, ml_output)

end subroutine mpas_atmphys_ml_convection_driver

!------------------------------------------------------------------
! SUBROUTINE: extract_column_for_ml
!
! Extract atmospheric column for ML input
! Includes resolution information for scale-aware prediction
!------------------------------------------------------------------
subroutine extract_column_for_ml(theta, qv, pressure, rho, &
                                  resolution, nz, state_vector)

    implicit none

    integer, intent(in) :: nz
    real(kind=RKIND), dimension(nz), intent(in) :: theta, qv, pressure, rho
    real(kind=RKIND), intent(in) :: resolution
    real(real32), intent(out) :: state_vector(:)

    integer :: k, idx

    idx = 1

    ! Pack vertical profiles
    do k = 1, nz
        state_vector(idx) = real(theta(k), real32)
        idx = idx + 1
    end do

    do k = 1, nz
        state_vector(idx) = real(qv(k), real32)
        idx = idx + 1
    end do

    ! Add local resolution (IMPORTANT for variable-resolution meshes!)
    state_vector(idx) = real(resolution, real32)
    idx = idx + 1

    ! Pad if necessary
    do while (idx <= ml_input_size)
        state_vector(idx) = 0.0
        idx = idx + 1
    end do

end subroutine extract_column_for_ml

!------------------------------------------------------------------
! SUBROUTINE: apply_ml_cu_tendencies
!
! Apply ML-predicted tendencies to MPAS state
!------------------------------------------------------------------
subroutine apply_ml_cu_tendencies(ml_output, tend_theta, tend_qv, nz, dt)

    implicit none

    integer, intent(in) :: nz
    real(real32), intent(in) :: ml_output(:)
    real(kind=RKIND), dimension(nz), intent(inout) :: tend_theta, tend_qv
    real(kind=RKIND), intent(in) :: dt

    integer :: k, idx

    idx = 1

    ! Temperature tendencies (K/s)
    do k = 1, nz
        tend_theta(k) = tend_theta(k) + real(ml_output(idx), RKIND)
        idx = idx + 1
    end do

    ! Moisture tendencies (kg/kg/s)
    do k = 1, nz
        tend_qv(k) = tend_qv(k) + real(ml_output(idx), RKIND)
        idx = idx + 1
    end do

end subroutine apply_ml_cu_tendencies

!------------------------------------------------------------------
! SUBROUTINE: mpas_atmphys_ml_convection_finalize
!
! Clean up ML convection scheme
!------------------------------------------------------------------
subroutine mpas_atmphys_ml_convection_finalize()

    implicit none

    if (ml_cu_initialized) then
        call torch_free_model(ml_cu_model)
        ml_cu_initialized = .false.
    end if

end subroutine mpas_atmphys_ml_convection_finalize

end module mpas_atmphys_ml_convection
```

### 2. Integrate into Physics Driver

**Edit `src/core_atmosphere/physics/mpas_atmphys_driver_convection.F`**:

```fortran
use mpas_atmphys_ml_convection  ! Add this

! In driver_convection subroutine:

! Add after existing convection schemes
if (config_conv_shallow_scheme == 'ml_convection') then
    call mpas_atmphys_ml_convection_driver(mesh, state, time_lev, &
                                            diag_physics, tend_physics, &
                                            configs, dt)
end if
```

### 3. Register in Registry

**Edit `src/core_atmosphere/Registry.xml`**:

```xml
<!-- ML Convection Configuration -->
<nml_option name="config_conv_shallow_scheme" type="character" default_value="off"
            units="unitless"
            description="Shallow convection scheme"
            possible_values="'off', 'camuwshcu', 'ml_convection'"/>

<nml_option name="config_ml_cu_model_path" type="character"
            default_value="convection_model.pt"
            units="unitless"
            description="Path to ML convection model"/>

<!-- ML tendency variables -->
<var name="ml_cu_tend_theta" type="real" dimensions="nVertLevels nCells Time"
     units="K s^{-1}"
     description="ML convection temperature tendency"/>
```

### 4. Update Namelist

**Edit `namelist.atmosphere`**:

```fortran
&physics
    config_conv_shallow_scheme = 'ml_convection'
    config_ml_cu_model_path = '/path/to/convection_model.pt'
/
```

## Example: Subgrid Turbulence Parameterization

```fortran
module mpas_atmphys_ml_turbulence

    use ftorch
    use mpas_kind_types
    use mpas_derived_types

    implicit none
    private
    public :: ml_turbulence_init, ml_turbulence_driver

    type(torch_model) :: turb_model

contains

subroutine ml_turbulence_driver(mesh, state, diag, tend, configs)

    ! This is particularly useful for variable-resolution meshes
    ! where turbulence scales vary with resolution

    type(mpas_pool_type), intent(in) :: mesh, state, configs
    type(mpas_pool_type), intent(inout) :: diag, tend

    integer, pointer :: nCells, nVertLevels
    real(kind=RKIND), dimension(:,:), pointer :: u, v, theta
    real(kind=RKIND), dimension(:), pointer :: meshDensity
    integer :: iCell

    call mpas_pool_get_dimension(mesh, 'nCells', nCells)
    call mpas_pool_get_array(mesh, 'meshDensity', meshDensity)

    do iCell = 1, nCells
        ! Extract turbulence-relevant quantities
        ! Include mesh resolution for scale-aware parameterization
        call compute_ml_turbulent_mixing(iCell, meshDensity(iCell), ...)
    end do

end subroutine ml_turbulence_driver

end module mpas_atmphys_ml_turbulence
```

## Handling Unstructured Grids

### Scale-Aware ML Models

Train models with resolution as input:

**Python Training**:

```python
import torch
import torch.nn as nn

class ScaleAwareConvection(nn.Module):
    def __init__(self):
        super().__init__()
        # Input includes resolution
        self.fc1 = nn.Linear(51, 128)  # 50 + 1 for resolution
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 20)

    def forward(self, x):
        # x[:, -1] is the grid resolution
        # Model can learn resolution-dependent behavior
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Training with variable resolution
for batch in dataloader:
    state, resolution, target = batch
    input = torch.cat([state, resolution.unsqueeze(1)], dim=1)
    output = model(input)
    loss = criterion(output, target)
```

### Mesh Connectivity (Advanced)

For schemes needing neighbor information:

```fortran
subroutine extract_cell_and_neighbors(iCell, mesh, state, ml_input)

    ! Get cell's neighbors on unstructured mesh
    integer, dimension(:,:), pointer :: cellsOnCell
    integer, dimension(:), pointer :: nEdgesOnCell

    call mpas_pool_get_array(mesh, 'cellsOnCell', cellsOnCell)
    call mpas_pool_get_array(mesh, 'nEdgesOnCell', nEdgesOnCell)

    ! Pack center cell state
    ! ...

    ! Pack neighbor states (for horizontal diffusion, advection)
    do i = 1, nEdgesOnCell(iCell)
        neighbor = cellsOnCell(i, iCell)
        ! Pack neighbor data
    end do

end subroutine extract_cell_and_neighbors
```

## Performance on Unstructured Mesh

### 1. Cell-by-Cell Processing

```fortran
! Standard approach - good for variable resolution
!$OMP PARALLEL DO PRIVATE(ml_input, ml_output, input_tensor, output_tensor)
do iCell = 1, nCells
    call extract_column(iCell, ml_input)
    input_tensor = torch_tensor_from_array(ml_input)
    output_tensor = torch_forward(model, input_tensor)
    call apply_tendencies(iCell, output_tensor)
    call torch_free_tensor(input_tensor)
    call torch_free_tensor(output_tensor)
end do
!$OMP END PARALLEL DO
```

### 2. Batched Processing (Advanced)

For regions of similar resolution:

```fortran
! Group cells by resolution
call group_cells_by_resolution(nCells, meshDensity, &
                                n_groups, cells_per_group)

do iGroup = 1, n_groups
    ! Batch process cells in same resolution range
    call batch_process_cells(cells_per_group(iGroup), ...)
end do
```

### 3. Load Balancing

MPAS uses domain decomposition. Ensure balanced ML load:

```fortran
! In mpas_atmphys_ml_convection_init
call mpas_dmpar_get_info(dminfo, nprocs=nProcs, my_proc_id=myProc)

! Monitor per-process timing
call mpas_timer_start('ml_convection')
! ... ML inference ...
call mpas_timer_stop('ml_convection')
```

## Testing and Validation

### 1. MPAS Test Cases

**Jablonowski-Williamson Baroclinic Wave**:

```bash
cd testing_and_setup/atmosphere/jw_baroclinic_wave
# Create mesh
./create_grid.sh
# Run init
./init_atmosphere_model
# Run model with ML
./atmosphere_model
```

**Supercell Test**:

```bash
cd testing_and_setup/atmosphere/supercell
# Similar workflow
```

### 2. Variable Resolution Testing

```bash
# Create variable resolution mesh
cd MPAS-Limited-Area
./create_region.py --refine-level 4 --refine-region convective_region
```

Verify ML scheme works across resolution ranges:

```python
import netCDF4 as nc
import numpy as np

# Read MPAS output
ds = nc.Dataset('history.*.nc')

# Get mesh density (inverse of cell area)
mesh_density = ds.variables['meshDensity'][:]
cell_area = 1.0 / mesh_density

# Plot ML tendencies vs resolution
ml_tend = ds.variables['ml_cu_tend_theta'][0, :, :]  # time, level, cell
plt.scatter(cell_area, ml_tend[:, 20])  # level 20
plt.xlabel('Cell Area (m^2)')
plt.ylabel('ML Heating Tendency (K/s)')
plt.xscale('log')
```

### 3. Parallel Scaling Test

```bash
# Test with different processor counts
for np in 4 8 16 32; do
    mpirun -n $np ./atmosphere_model
    # Extract timing from log
done

# Plot scaling
python plot_scaling.py
```

## Best Practices for MPAS

1. **Include Resolution**: Always pass mesh resolution to ML models
2. **Thread Safety**: Each OpenMP thread needs its own tensors
3. **MPI Safety**: Each MPI rank loads its own model (no cross-rank tensor sharing)
4. **Timing**: Use MPAS timer infrastructure
5. **Registry**: Register all ML variables properly
6. **Validation**: Test on both uniform and variable resolution meshes

## Troubleshooting

**Issue: Performance degradation with variable resolution**

```fortran
! Solution: Batch cells by resolution
! Group similar-resolution cells together for batched inference
```

**Issue: MPI hangs**

```fortran
! Ensure all ranks load model
call mpas_dmpar_barrier(dminfo)
call ml_convection_init(...)
call mpas_dmpar_barrier(dminfo)
```

**Issue: Memory usage scales with processor count**

```fortran
! Expected - each MPI rank loads its own model
! Monitor with: /usr/bin/time -v mpirun -n 16 ./atmosphere_model
```

## Summary

Integration into MPAS requires:

1. ✓ Build Fortran-Torch and link with MPAS
2. ✓ Create ML physics module
3. ✓ Handle unstructured mesh (pass resolution to ML)
4. ✓ Register in Registry.xml
5. ✓ Update physics driver
6. ✓ Test on variable resolution meshes

Key advantage: ML models can be **scale-aware**, learning different behavior at different resolutions!

## References

- MPAS Documentation: https://mpas-dev.github.io/
- MPAS-A User's Guide: https://www2.mmm.ucar.edu/projects/mpas/mpas_atmosphere_users_guide.html
- Fortran-Torch: https://github.com/fzhao70/Fortran-Torch

---

For questions, see [GitHub Issues](https://github.com/fzhao70/Fortran-Torch/issues).

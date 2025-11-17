program weather_model_example
    !> Weather Model Integration Example
    !>
    !> This example demonstrates how to integrate a PyTorch neural network
    !> into a Fortran-based weather model for parameterization.
    !>
    !> Simulates a simple scenario where:
    !> - We have atmospheric state variables at grid points
    !> - The NN predicts parameterized tendencies
    !> - We apply the tendencies to update the state

    use ftorch
    use iso_fortran_env, only: real32, real64
    implicit none

    ! Model grid parameters
    integer, parameter :: nx = 10           ! Number of x grid points
    integer, parameter :: ny = 10           ! Number of y grid points
    integer, parameter :: nz = 5            ! Number of vertical levels
    integer, parameter :: n_state_vars = 2  ! Temperature and humidity

    ! Neural network parameters
    integer, parameter :: nn_input_size = 50
    integer, parameter :: nn_output_size = 20

    ! Variables
    type(torch_model) :: nn_model
    type(torch_tensor) :: input_tensor, output_tensor

    real(real32) :: temperature(nx, ny, nz)
    real(real32) :: humidity(nx, ny, nz)
    real(real32) :: nn_input(nn_input_size)
    real(real32) :: nn_output(nn_output_size)

    integer :: i, j, k, timestep
    integer, parameter :: n_timesteps = 5
    real(real64) :: start_time, end_time, nn_time

    print *, ''
    print *, '============================================================'
    print *, 'Weather Model Integration Example'
    print *, '============================================================'
    print *, ''

    ! Initialize the model state
    call initialize_atmosphere(temperature, humidity)

    ! Load the neural network model
    print *, 'Loading neural network parameterization...'
    nn_model = torch_load_model('weather_model.pt', TORCH_DEVICE_CPU)

    if (.not. c_associated(nn_model%ptr)) then
        print *, 'Error: Failed to load weather_model.pt'
        print *, 'Please run: python examples/python/weather_model.py'
        stop 1
    end if

    print *, 'Neural network loaded successfully!'
    print *, ''

    ! Time integration loop
    print *, 'Starting time integration...'
    print *, ''

    nn_time = 0.0

    do timestep = 1, n_timesteps
        print *, 'Timestep: ', timestep
        print *, '  Mean temperature: ', sum(temperature) / size(temperature)
        print *, '  Mean humidity:    ', sum(humidity) / size(humidity)

        ! Loop over grid points
        do j = 1, ny
            do i = 1, nx
                ! Extract atmospheric state at this grid column
                call extract_column_state(temperature, humidity, i, j, nn_input)

                ! Create tensor from input
                input_tensor = torch_tensor_from_array(nn_input)

                ! Run neural network inference
                call cpu_time(start_time)
                output_tensor = torch_forward(nn_model, input_tensor)
                call cpu_time(end_time)
                nn_time = nn_time + (end_time - start_time)

                ! Extract output
                call torch_tensor_to_array(output_tensor, nn_output)

                ! Apply parameterized tendencies
                call apply_tendencies(temperature, humidity, i, j, nn_output)

                ! Cleanup tensors
                call torch_free_tensor(input_tensor)
                call torch_free_tensor(output_tensor)
            end do
        end do

        print *, ''
    end do

    print *, 'Time integration completed!'
    print *, ''
    print *, 'Performance statistics:'
    print *, '  Total NN inference time: ', nn_time, ' seconds'
    print *, '  Number of NN calls:      ', nx * ny * n_timesteps
    print *, '  Average time per call:   ', nn_time / (nx * ny * n_timesteps) * 1000.0, ' ms'
    print *, ''

    ! Final state
    print *, 'Final state:'
    print *, '  Mean temperature: ', sum(temperature) / size(temperature)
    print *, '  Mean humidity:    ', sum(humidity) / size(humidity)
    print *, ''

    ! Cleanup
    call torch_free_model(nn_model)

    print *, '============================================================'
    print *, 'Example completed successfully!'
    print *, '============================================================'
    print *, ''

contains

    subroutine initialize_atmosphere(temp, hum)
        !> Initialize atmospheric state with realistic values
        real(real32), intent(out) :: temp(:,:,:)
        real(real32), intent(out) :: hum(:,:,:)
        integer :: i, j, k
        real(real32) :: height_factor

        do k = 1, nz
            height_factor = real(k, real32) / real(nz, real32)

            do j = 1, ny
                do i = 1, nx
                    ! Temperature decreases with height
                    ! Add some spatial variability
                    temp(i, j, k) = 288.0 - 6.5 * height_factor * 10.0 &
                                  + 5.0 * sin(real(i, real32) * 0.5) &
                                  * cos(real(j, real32) * 0.3)

                    ! Humidity decreases with height
                    hum(i, j, k) = 0.8 * (1.0 - height_factor) &
                                 + 0.1 * cos(real(i, real32) * 0.4) &
                                 * sin(real(j, real32) * 0.6)
                end do
            end do
        end do
    end subroutine initialize_atmosphere

    subroutine extract_column_state(temp, hum, i, j, state_vector)
        !> Extract atmospheric state at a grid column for NN input
        real(real32), intent(in) :: temp(:,:,:)
        real(real32), intent(in) :: hum(:,:,:)
        integer, intent(in) :: i, j
        real(real32), intent(out) :: state_vector(:)

        integer :: k, idx

        ! Pack temperature and humidity at all levels
        idx = 1
        do k = 1, nz
            state_vector(idx) = temp(i, j, k)
            idx = idx + 1
        end do

        do k = 1, nz
            state_vector(idx) = hum(i, j, k)
            idx = idx + 1
        end do

        ! Add derived quantities or other variables
        ! (normalized height, pressure, etc.)
        do k = 1, nz
            state_vector(idx) = real(k, real32) / real(nz, real32)
            idx = idx + 1
        end do

        ! Pad remaining with zeros or additional features
        do while (idx <= nn_input_size)
            state_vector(idx) = 0.0
            idx = idx + 1
        end do
    end subroutine extract_column_state

    subroutine apply_tendencies(temp, hum, i, j, tendencies)
        !> Apply NN-predicted tendencies to the atmospheric state
        real(real32), intent(inout) :: temp(:,:,:)
        real(real32), intent(inout) :: hum(:,:,:)
        integer, intent(in) :: i, j
        real(real32), intent(in) :: tendencies(:)

        integer :: k, idx
        real(real32), parameter :: dt = 100.0  ! Time step in seconds
        real(real32), parameter :: tendency_scale = 0.01

        ! Apply temperature tendencies
        idx = 1
        do k = 1, nz
            if (idx <= nn_output_size) then
                temp(i, j, k) = temp(i, j, k) + tendencies(idx) * dt * tendency_scale
                idx = idx + 1
            end if
        end do

        ! Apply humidity tendencies
        do k = 1, nz
            if (idx <= nn_output_size) then
                hum(i, j, k) = hum(i, j, k) + tendencies(idx) * dt * tendency_scale
                ! Ensure humidity stays in valid range [0, 1]
                hum(i, j, k) = max(0.0, min(1.0, hum(i, j, k)))
                idx = idx + 1
            end if
        end do
    end subroutine apply_tendencies

end program weather_model_example

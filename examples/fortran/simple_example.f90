program simple_example
    !> Simple example demonstrating the Fortran-Torch framework
    !>
    !> This program shows how to:
    !> 1. Load a TorchScript model
    !> 2. Create input tensors from Fortran arrays
    !> 3. Run inference
    !> 4. Extract results back to Fortran arrays
    !> 5. Clean up resources

    use ftorch
    use iso_fortran_env, only: real32
    implicit none

    ! Variables
    type(torch_model) :: model
    type(torch_tensor) :: input_tensor, output_tensor
    real(real32), dimension(10) :: input_data
    real(real32), dimension(5) :: output_data
    integer :: i
    logical :: cuda_available

    print *, '========================================='
    print *, 'Fortran-Torch Simple Example'
    print *, '========================================='
    print *, ''

    ! Check CUDA availability
    cuda_available = torch_cuda_available()
    print *, 'CUDA available: ', cuda_available
    print *, ''

    ! Step 1: Load the model
    print *, 'Step 1: Loading TorchScript model...'
    model = torch_load_model('simple_model.pt', TORCH_DEVICE_CPU)

    if (.not. c_associated(model%ptr)) then
        print *, 'Error: Failed to load model!'
        print *, 'Make sure to run the Python script first:'
        print *, '  python examples/python/simple_model.py'
        stop 1
    end if

    print *, 'Model loaded successfully!'
    print *, ''

    ! Step 2: Prepare input data
    print *, 'Step 2: Preparing input data...'
    do i = 1, 10
        input_data(i) = real(i, real32) * 0.1
    end do

    print *, 'Input data:'
    print '(10F8.3)', input_data
    print *, ''

    ! Step 3: Create input tensor
    print *, 'Step 3: Creating input tensor...'
    input_tensor = torch_tensor_from_array(input_data)

    if (.not. c_associated(input_tensor%ptr)) then
        print *, 'Error: Failed to create input tensor!'
        call torch_free_model(model)
        stop 1
    end if

    print *, 'Input tensor created!'
    print *, ''

    ! Step 4: Run inference
    print *, 'Step 4: Running inference...'
    output_tensor = torch_forward(model, input_tensor)

    if (.not. c_associated(output_tensor%ptr)) then
        print *, 'Error: Forward pass failed!'
        call torch_free_tensor(input_tensor)
        call torch_free_model(model)
        stop 1
    end if

    print *, 'Inference completed!'
    print *, ''

    ! Step 5: Extract output data
    print *, 'Step 5: Extracting output data...'
    call torch_tensor_to_array(output_tensor, output_data)

    print *, 'Output data:'
    print '(5F10.4)', output_data
    print *, ''

    ! Step 6: Cleanup
    print *, 'Step 6: Cleaning up...'
    call torch_free_tensor(input_tensor)
    call torch_free_tensor(output_tensor)
    call torch_free_model(model)

    print *, 'Done!'
    print *, ''
    print *, '========================================='
    print *, 'Example completed successfully!'
    print *, '========================================='

end program simple_example

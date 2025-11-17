program test_inference
    !> End-to-end inference test
    !>
    !> This test requires a trained model file (simple_model.pt)
    !> Create it by running: python examples/python/simple_model.py

    use ftorch
    use iso_fortran_env, only: real32
    implicit none

    type(torch_model) :: model
    type(torch_tensor) :: input_tensor, output_tensor
    real(real32) :: input_data(10)
    real(real32) :: output_data(5)
    logical :: file_exists
    integer :: i

    print *, ''
    print *, '======================================'
    print *, 'Fortran-Torch Inference Test'
    print *, '======================================'
    print *, ''

    ! Check if model exists
    inquire(file='simple_model.pt', exist=file_exists)

    if (.not. file_exists) then
        print *, 'ERROR: simple_model.pt not found'
        print *, ''
        print *, 'Please create the model first by running:'
        print *, '  cd examples/python'
        print *, '  python simple_model.py'
        print *, '  cp simple_model.pt ../../tests/fortran/'
        print *, ''
        stop 1
    end if

    print *, 'Step 1: Loading model...'
    model = torch_load_model('simple_model.pt', TORCH_DEVICE_CPU)

    if (.not. c_associated(model%ptr)) then
        print *, 'FAILED: Model loading failed'
        stop 1
    end if
    print *, '  Model loaded successfully'
    print *, ''

    print *, 'Step 2: Preparing input data...'
    do i = 1, 10
        input_data(i) = real(i, real32) * 0.1
    end do
    print *, '  Input data:'
    print '(10F8.3)', input_data
    print *, ''

    print *, 'Step 3: Creating input tensor...'
    input_tensor = torch_tensor_from_array(input_data)

    if (.not. c_associated(input_tensor%ptr)) then
        print *, 'FAILED: Tensor creation failed'
        call torch_free_model(model)
        stop 1
    end if
    print *, '  Tensor created'
    print *, ''

    print *, 'Step 4: Running inference...'
    output_tensor = torch_forward(model, input_tensor)

    if (.not. c_associated(output_tensor%ptr)) then
        print *, 'FAILED: Inference failed'
        call torch_free_tensor(input_tensor)
        call torch_free_model(model)
        stop 1
    end if
    print *, '  Inference completed'
    print *, ''

    print *, 'Step 5: Extracting output...'
    call torch_tensor_to_array(output_tensor, output_data)
    print *, '  Output data:'
    print '(5F10.4)', output_data
    print *, ''

    print *, 'Step 6: Running multiple inferences (performance test)...'
    do i = 1, 100
        call torch_free_tensor(input_tensor)
        call torch_free_tensor(output_tensor)

        input_tensor = torch_tensor_from_array(input_data)
        output_tensor = torch_forward(model, input_tensor)
    end do
    print *, '  100 inferences completed'
    print *, ''

    print *, 'Step 7: Cleanup...'
    call torch_free_tensor(input_tensor)
    call torch_free_tensor(output_tensor)
    call torch_free_model(model)
    print *, '  Resources freed'
    print *, ''

    print *, '======================================'
    print *, 'ALL TESTS PASSED'
    print *, '======================================'
    print *, ''

end program test_inference

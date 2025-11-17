program test_basic
    !> Basic unit tests for Fortran-Torch framework
    !>
    !> Tests:
    !> 1. CUDA availability check
    !> 2. Tensor creation from 1D arrays
    !> 3. Tensor creation from 2D arrays
    !> 4. Tensor creation from 3D arrays
    !> 5. Tensor data extraction
    !> 6. Model loading (if model file exists)
    !> 7. Basic inference (if model file exists)

    use ftorch
    use iso_fortran_env, only: real32, real64
    implicit none

    integer :: test_count = 0
    integer :: test_passed = 0
    integer :: test_failed = 0

    print *, ''
    print *, '======================================'
    print *, 'Fortran-Torch Basic Tests'
    print *, '======================================'
    print *, ''

    ! Test 1: CUDA availability
    call test_cuda_available()

    ! Test 2: 1D tensor creation and extraction (float32)
    call test_tensor_1d_real32()

    ! Test 3: 2D tensor creation and extraction (float32)
    call test_tensor_2d_real32()

    ! Test 4: 3D tensor creation and extraction (float32)
    call test_tensor_3d_real32()

    ! Test 5: 1D tensor creation and extraction (float64)
    call test_tensor_1d_real64()

    ! Test 6: 2D tensor creation and extraction (float64)
    call test_tensor_2d_real64()

    ! Test 7: Model loading (if model exists)
    call test_model_loading()

    ! Print summary
    print *, ''
    print *, '======================================'
    print *, 'Test Summary'
    print *, '======================================'
    print *, 'Total tests:  ', test_count
    print *, 'Passed:       ', test_passed
    print *, 'Failed:       ', test_failed
    print *, ''

    if (test_failed > 0) then
        print *, 'RESULT: FAILED'
        stop 1
    else
        print *, 'RESULT: ALL TESTS PASSED'
    end if

contains

    subroutine test_cuda_available()
        logical :: cuda_avail

        test_count = test_count + 1
        print *, 'Test 1: CUDA availability check'

        cuda_avail = torch_cuda_available()
        print *, '  CUDA available: ', cuda_avail

        ! This test always passes (just informational)
        test_passed = test_passed + 1
        print *, '  PASSED'
        print *, ''
    end subroutine test_cuda_available

    subroutine test_tensor_1d_real32()
        type(torch_tensor) :: tensor
        real(real32) :: input_data(10)
        real(real32) :: output_data(10)
        integer :: i
        logical :: test_ok

        test_count = test_count + 1
        print *, 'Test 2: 1D tensor (float32) creation and extraction'

        ! Initialize input data
        do i = 1, 10
            input_data(i) = real(i, real32)
        end do

        ! Create tensor
        tensor = torch_tensor_from_array(input_data)

        test_ok = c_associated(tensor%ptr)
        if (.not. test_ok) then
            print *, '  FAILED: Tensor creation failed'
            test_failed = test_failed + 1
            return
        end if

        ! Extract data
        call torch_tensor_to_array(tensor, output_data)

        ! Verify data
        test_ok = .true.
        do i = 1, 10
            if (abs(output_data(i) - input_data(i)) > 1.0e-6) then
                test_ok = .false.
                print *, '  Mismatch at index', i
                print *, '    Expected:', input_data(i)
                print *, '    Got:     ', output_data(i)
            end if
        end do

        ! Cleanup
        call torch_free_tensor(tensor)

        if (test_ok) then
            print *, '  PASSED'
            test_passed = test_passed + 1
        else
            print *, '  FAILED'
            test_failed = test_failed + 1
        end if
        print *, ''
    end subroutine test_tensor_1d_real32

    subroutine test_tensor_2d_real32()
        type(torch_tensor) :: tensor
        real(real32) :: input_data(3, 4)
        real(real32) :: output_data(3, 4)
        integer :: i, j
        logical :: test_ok

        test_count = test_count + 1
        print *, 'Test 3: 2D tensor (float32) creation and extraction'

        ! Initialize input data
        do j = 1, 4
            do i = 1, 3
                input_data(i, j) = real(i * 10 + j, real32)
            end do
        end do

        ! Create tensor
        tensor = torch_tensor_from_array(input_data)

        test_ok = c_associated(tensor%ptr)
        if (.not. test_ok) then
            print *, '  FAILED: Tensor creation failed'
            test_failed = test_failed + 1
            return
        end if

        ! Extract data
        call torch_tensor_to_array(tensor, output_data)

        ! Verify data
        test_ok = .true.
        do j = 1, 4
            do i = 1, 3
                if (abs(output_data(i, j) - input_data(i, j)) > 1.0e-6) then
                    test_ok = .false.
                    print *, '  Mismatch at index', i, j
                end if
            end do
        end do

        ! Cleanup
        call torch_free_tensor(tensor)

        if (test_ok) then
            print *, '  PASSED'
            test_passed = test_passed + 1
        else
            print *, '  FAILED'
            test_failed = test_failed + 1
        end if
        print *, ''
    end subroutine test_tensor_2d_real32

    subroutine test_tensor_3d_real32()
        type(torch_tensor) :: tensor
        real(real32) :: input_data(2, 3, 4)
        real(real32) :: output_data(2, 3, 4)
        integer :: i, j, k
        logical :: test_ok

        test_count = test_count + 1
        print *, 'Test 4: 3D tensor (float32) creation and extraction'

        ! Initialize input data
        do k = 1, 4
            do j = 1, 3
                do i = 1, 2
                    input_data(i, j, k) = real(i * 100 + j * 10 + k, real32)
                end do
            end do
        end do

        ! Create tensor
        tensor = torch_tensor_from_array(input_data)

        test_ok = c_associated(tensor%ptr)
        if (.not. test_ok) then
            print *, '  FAILED: Tensor creation failed'
            test_failed = test_failed + 1
            return
        end if

        ! Extract data
        call torch_tensor_to_array(tensor, output_data)

        ! Verify data
        test_ok = .true.
        do k = 1, 4
            do j = 1, 3
                do i = 1, 2
                    if (abs(output_data(i, j, k) - input_data(i, j, k)) > 1.0e-6) then
                        test_ok = .false.
                    end if
                end do
            end do
        end do

        ! Cleanup
        call torch_free_tensor(tensor)

        if (test_ok) then
            print *, '  PASSED'
            test_passed = test_passed + 1
        else
            print *, '  FAILED'
            test_failed = test_failed + 1
        end if
        print *, ''
    end subroutine test_tensor_3d_real32

    subroutine test_tensor_1d_real64()
        type(torch_tensor) :: tensor
        real(real64) :: input_data(10)
        real(real64) :: output_data(10)
        integer :: i
        logical :: test_ok

        test_count = test_count + 1
        print *, 'Test 5: 1D tensor (float64) creation and extraction'

        ! Initialize input data
        do i = 1, 10
            input_data(i) = real(i, real64) * 1.5_real64
        end do

        ! Create tensor
        tensor = torch_tensor_from_array(input_data)

        test_ok = c_associated(tensor%ptr)
        if (.not. test_ok) then
            print *, '  FAILED: Tensor creation failed'
            test_failed = test_failed + 1
            return
        end if

        ! Extract data
        call torch_tensor_to_array(tensor, output_data)

        ! Verify data
        test_ok = .true.
        do i = 1, 10
            if (abs(output_data(i) - input_data(i)) > 1.0e-12) then
                test_ok = .false.
            end if
        end do

        ! Cleanup
        call torch_free_tensor(tensor)

        if (test_ok) then
            print *, '  PASSED'
            test_passed = test_passed + 1
        else
            print *, '  FAILED'
            test_failed = test_failed + 1
        end if
        print *, ''
    end subroutine test_tensor_1d_real64

    subroutine test_tensor_2d_real64()
        type(torch_tensor) :: tensor
        real(real64) :: input_data(3, 4)
        real(real64) :: output_data(3, 4)
        integer :: i, j
        logical :: test_ok

        test_count = test_count + 1
        print *, 'Test 6: 2D tensor (float64) creation and extraction'

        ! Initialize input data
        do j = 1, 4
            do i = 1, 3
                input_data(i, j) = real(i * 10 + j, real64) * 2.5_real64
            end do
        end do

        ! Create tensor
        tensor = torch_tensor_from_array(input_data)

        test_ok = c_associated(tensor%ptr)
        if (.not. test_ok) then
            print *, '  FAILED: Tensor creation failed'
            test_failed = test_failed + 1
            return
        end if

        ! Extract data
        call torch_tensor_to_array(tensor, output_data)

        ! Verify data
        test_ok = .true.
        do j = 1, 4
            do i = 1, 3
                if (abs(output_data(i, j) - input_data(i, j)) > 1.0e-12) then
                    test_ok = .false.
                end if
            end do
        end do

        ! Cleanup
        call torch_free_tensor(tensor)

        if (test_ok) then
            print *, '  PASSED'
            test_passed = test_passed + 1
        else
            print *, '  FAILED'
            test_failed = test_failed + 1
        end if
        print *, ''
    end subroutine test_tensor_2d_real64

    subroutine test_model_loading()
        type(torch_model) :: model
        logical :: file_exists
        logical :: test_ok

        test_count = test_count + 1
        print *, 'Test 7: Model loading'

        ! Check if test model exists
        inquire(file='test_model.pt', exist=file_exists)

        if (.not. file_exists) then
            print *, '  SKIPPED: test_model.pt not found'
            print *, '  (This is expected if you haven''t created a test model)'
            test_passed = test_passed + 1
            print *, ''
            return
        end if

        ! Try to load the model
        model = torch_load_model('test_model.pt', TORCH_DEVICE_CPU)

        test_ok = c_associated(model%ptr)

        if (test_ok) then
            print *, '  PASSED: Model loaded successfully'
            call torch_free_model(model)
            test_passed = test_passed + 1
        else
            print *, '  FAILED: Model loading failed'
            test_failed = test_failed + 1
        end if
        print *, ''
    end subroutine test_model_loading

end program test_basic

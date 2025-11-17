module ftorch
    use, intrinsic :: iso_c_binding
    implicit none

    private
    public :: torch_model, torch_tensor
    public :: torch_dtype, torch_device
    public :: torch_load_model, torch_free_model
    public :: torch_tensor_from_array, torch_tensor_to_array
    public :: torch_forward, torch_forward_multi
    public :: torch_free_tensor
    public :: torch_cuda_available
    public :: TORCH_FLOAT32, TORCH_FLOAT64, TORCH_INT32, TORCH_INT64
    public :: TORCH_DEVICE_CPU, TORCH_DEVICE_CUDA

    ! Opaque types
    type :: torch_model
        type(c_ptr) :: ptr = c_null_ptr
    end type torch_model

    type :: torch_tensor
        type(c_ptr) :: ptr = c_null_ptr
    end type torch_tensor

    ! Data types
    enum, bind(c)
        enumerator :: TORCH_FLOAT32 = 0
        enumerator :: TORCH_FLOAT64 = 1
        enumerator :: TORCH_INT32 = 2
        enumerator :: TORCH_INT64 = 3
    end enum

    ! Device types
    enum, bind(c)
        enumerator :: TORCH_DEVICE_CPU = 0
        enumerator :: TORCH_DEVICE_CUDA = 1
    end enum

    integer, parameter :: torch_dtype = c_int
    integer, parameter :: torch_device = c_int

    ! C interface
    interface
        function ftorch_load_model_c(model_path, device, model) result(status) bind(c, name='ftorch_load_model')
            import :: c_char, c_int, c_ptr
            character(kind=c_char), dimension(*), intent(in) :: model_path
            integer(c_int), value :: device
            type(c_ptr), intent(out) :: model
            integer(c_int) :: status
        end function ftorch_load_model_c

        subroutine ftorch_free_model_c(model) bind(c, name='ftorch_free_model')
            import :: c_ptr
            type(c_ptr), value :: model
        end subroutine ftorch_free_model_c

        function ftorch_create_tensor_c(data, ndim, shape, dtype, device, tensor) result(status) &
            bind(c, name='ftorch_create_tensor')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: data
            integer(c_int), value :: ndim
            type(c_ptr), value :: shape
            integer(c_int), value :: dtype
            integer(c_int), value :: device
            type(c_ptr), intent(out) :: tensor
            integer(c_int) :: status
        end function ftorch_create_tensor_c

        function ftorch_tensor_to_array_c(tensor, data) result(status) bind(c, name='ftorch_tensor_to_array')
            import :: c_ptr, c_int
            type(c_ptr), value :: tensor
            type(c_ptr), value :: data
            integer(c_int) :: status
        end function ftorch_tensor_to_array_c

        function ftorch_tensor_shape_c(tensor, ndim, shape) result(status) bind(c, name='ftorch_tensor_shape')
            import :: c_ptr, c_int
            type(c_ptr), value :: tensor
            type(c_ptr), value :: ndim
            type(c_ptr), value :: shape
            integer(c_int) :: status
        end function ftorch_tensor_shape_c

        subroutine ftorch_free_tensor_c(tensor) bind(c, name='ftorch_free_tensor')
            import :: c_ptr
            type(c_ptr), value :: tensor
        end subroutine ftorch_free_tensor_c

        function ftorch_forward_c(model, input, output) result(status) bind(c, name='ftorch_forward')
            import :: c_ptr, c_int
            type(c_ptr), value :: model
            type(c_ptr), value :: input
            type(c_ptr), intent(out) :: output
            integer(c_int) :: status
        end function ftorch_forward_c

        function ftorch_forward_multi_c(model, inputs, n_inputs, output) result(status) &
            bind(c, name='ftorch_forward_multi')
            import :: c_ptr, c_int
            type(c_ptr), value :: model
            type(c_ptr), value :: inputs
            integer(c_int), value :: n_inputs
            type(c_ptr), intent(out) :: output
            integer(c_int) :: status
        end function ftorch_forward_multi_c

        function ftorch_cuda_available_c() result(available) bind(c, name='ftorch_cuda_available')
            import :: c_int
            integer(c_int) :: available
        end function ftorch_cuda_available_c

        function ftorch_get_last_error_c() result(msg) bind(c, name='ftorch_get_last_error')
            import :: c_ptr
            type(c_ptr) :: msg
        end function ftorch_get_last_error_c
    end interface

    ! Fortran wrapper interfaces
    interface torch_tensor_from_array
        module procedure torch_tensor_from_array_real32_1d
        module procedure torch_tensor_from_array_real32_2d
        module procedure torch_tensor_from_array_real32_3d
        module procedure torch_tensor_from_array_real64_1d
        module procedure torch_tensor_from_array_real64_2d
        module procedure torch_tensor_from_array_real64_3d
    end interface torch_tensor_from_array

    interface torch_tensor_to_array
        module procedure torch_tensor_to_array_real32_1d
        module procedure torch_tensor_to_array_real32_2d
        module procedure torch_tensor_to_array_real32_3d
        module procedure torch_tensor_to_array_real64_1d
        module procedure torch_tensor_to_array_real64_2d
        module procedure torch_tensor_to_array_real64_3d
    end interface torch_tensor_to_array

contains

    !> Load a TorchScript model
    function torch_load_model(model_path, device) result(model)
        character(len=*), intent(in) :: model_path
        integer(torch_device), intent(in), optional :: device
        type(torch_model) :: model

        integer(c_int) :: status
        integer(torch_device) :: dev

        if (present(device)) then
            dev = device
        else
            dev = TORCH_DEVICE_CPU
        end if

        status = ftorch_load_model_c(trim(model_path) // c_null_char, dev, model%ptr)

        if (status /= 0) then
            write(*,'(A,I0)') 'Error loading model, status: ', status
            call print_last_error()
        end if
    end function torch_load_model

    !> Free a model
    subroutine torch_free_model(model)
        type(torch_model), intent(inout) :: model
        call ftorch_free_model_c(model%ptr)
        model%ptr = c_null_ptr
    end subroutine torch_free_model

    !> Free a tensor
    subroutine torch_free_tensor(tensor)
        type(torch_tensor), intent(inout) :: tensor
        call ftorch_free_tensor_c(tensor%ptr)
        tensor%ptr = c_null_ptr
    end subroutine torch_free_tensor

    !> Run forward pass
    function torch_forward(model, input) result(output)
        type(torch_model), intent(in) :: model
        type(torch_tensor), intent(in) :: input
        type(torch_tensor) :: output

        integer(c_int) :: status

        status = ftorch_forward_c(model%ptr, input%ptr, output%ptr)

        if (status /= 0) then
            write(*,'(A,I0)') 'Error in forward pass, status: ', status
            call print_last_error()
        end if
    end function torch_forward

    !> Run forward pass with multiple inputs
    function torch_forward_multi(model, inputs) result(output)
        type(torch_model), intent(in) :: model
        type(torch_tensor), intent(in) :: inputs(:)
        type(torch_tensor) :: output

        integer(c_int) :: status
        type(c_ptr), allocatable :: input_ptrs(:)
        integer :: i

        allocate(input_ptrs(size(inputs)))
        do i = 1, size(inputs)
            input_ptrs(i) = inputs(i)%ptr
        end do

        status = ftorch_forward_multi_c(model%ptr, c_loc(input_ptrs), size(inputs), output%ptr)

        if (status /= 0) then
            write(*,'(A,I0)') 'Error in forward pass, status: ', status
            call print_last_error()
        end if

        deallocate(input_ptrs)
    end function torch_forward_multi

    !> Check if CUDA is available
    function torch_cuda_available() result(available)
        logical :: available
        available = (ftorch_cuda_available_c() /= 0)
    end function torch_cuda_available

    !> Print last error message
    subroutine print_last_error()
        type(c_ptr) :: msg_ptr
        character(len=:), allocatable :: error_msg
        integer :: i, msg_len

        msg_ptr = ftorch_get_last_error_c()
        if (.not. c_associated(msg_ptr)) return

        ! Find string length
        msg_len = 0
        do i = 0, 1000
            if (get_char_at(msg_ptr, i) == c_null_char) exit
            msg_len = msg_len + 1
        end do

        ! Copy string
        allocate(character(len=msg_len) :: error_msg)
        do i = 1, msg_len
            error_msg(i:i) = get_char_at(msg_ptr, i-1)
        end do

        write(*,'(A,A)') 'PyTorch error: ', error_msg
    end subroutine print_last_error

    function get_char_at(ptr, offset) result(ch)
        type(c_ptr), intent(in) :: ptr
        integer, intent(in) :: offset
        character(len=1) :: ch
        character(len=1), pointer :: fptr
        type(c_ptr) :: offset_ptr

        offset_ptr = c_ptr_plus_offset(ptr, offset)
        call c_f_pointer(offset_ptr, fptr)
        ch = fptr
    end function get_char_at

    function c_ptr_plus_offset(ptr, offset) result(new_ptr)
        type(c_ptr), intent(in) :: ptr
        integer, intent(in) :: offset
        type(c_ptr) :: new_ptr
        integer(c_intptr_t) :: addr

        addr = transfer(ptr, addr)
        addr = addr + offset
        new_ptr = transfer(addr, new_ptr)
    end function c_ptr_plus_offset

    ! Tensor creation for real32 arrays
    function torch_tensor_from_array_real32_1d(array, device) result(tensor)
        real(c_float), target, intent(in) :: array(:)
        integer(torch_device), intent(in), optional :: device
        type(torch_tensor) :: tensor

        integer(c_int64_t) :: shape(1)
        integer(torch_device) :: dev
        integer(c_int) :: status

        shape(1) = size(array, 1)
        dev = TORCH_DEVICE_CPU
        if (present(device)) dev = device

        status = ftorch_create_tensor_c(c_loc(array), 1, c_loc(shape), &
                                        TORCH_FLOAT32, dev, tensor%ptr)

        if (status /= 0) then
            write(*,'(A,I0)') 'Error creating tensor, status: ', status
            call print_last_error()
        end if
    end function torch_tensor_from_array_real32_1d

    function torch_tensor_from_array_real32_2d(array, device) result(tensor)
        real(c_float), target, intent(in) :: array(:,:)
        integer(torch_device), intent(in), optional :: device
        type(torch_tensor) :: tensor

        integer(c_int64_t) :: shape(2)
        integer(torch_device) :: dev
        integer(c_int) :: status

        shape(1) = size(array, 1)
        shape(2) = size(array, 2)
        dev = TORCH_DEVICE_CPU
        if (present(device)) dev = device

        status = ftorch_create_tensor_c(c_loc(array), 2, c_loc(shape), &
                                        TORCH_FLOAT32, dev, tensor%ptr)

        if (status /= 0) then
            write(*,'(A,I0)') 'Error creating tensor, status: ', status
            call print_last_error()
        end if
    end function torch_tensor_from_array_real32_2d

    function torch_tensor_from_array_real32_3d(array, device) result(tensor)
        real(c_float), target, intent(in) :: array(:,:,:)
        integer(torch_device), intent(in), optional :: device
        type(torch_tensor) :: tensor

        integer(c_int64_t) :: shape(3)
        integer(torch_device) :: dev
        integer(c_int) :: status

        shape(1) = size(array, 1)
        shape(2) = size(array, 2)
        shape(3) = size(array, 3)
        dev = TORCH_DEVICE_CPU
        if (present(device)) dev = device

        status = ftorch_create_tensor_c(c_loc(array), 3, c_loc(shape), &
                                        TORCH_FLOAT32, dev, tensor%ptr)

        if (status /= 0) then
            write(*,'(A,I0)') 'Error creating tensor, status: ', status
            call print_last_error()
        end if
    end function torch_tensor_from_array_real32_3d

    ! Tensor creation for real64 arrays
    function torch_tensor_from_array_real64_1d(array, device) result(tensor)
        real(c_double), target, intent(in) :: array(:)
        integer(torch_device), intent(in), optional :: device
        type(torch_tensor) :: tensor

        integer(c_int64_t) :: shape(1)
        integer(torch_device) :: dev
        integer(c_int) :: status

        shape(1) = size(array, 1)
        dev = TORCH_DEVICE_CPU
        if (present(device)) dev = device

        status = ftorch_create_tensor_c(c_loc(array), 1, c_loc(shape), &
                                        TORCH_FLOAT64, dev, tensor%ptr)

        if (status /= 0) then
            write(*,'(A,I0)') 'Error creating tensor, status: ', status
            call print_last_error()
        end if
    end function torch_tensor_from_array_real64_1d

    function torch_tensor_from_array_real64_2d(array, device) result(tensor)
        real(c_double), target, intent(in) :: array(:,:)
        integer(torch_device), intent(in), optional :: device
        type(torch_tensor) :: tensor

        integer(c_int64_t) :: shape(2)
        integer(torch_device) :: dev
        integer(c_int) :: status

        shape(1) = size(array, 1)
        shape(2) = size(array, 2)
        dev = TORCH_DEVICE_CPU
        if (present(device)) dev = device

        status = ftorch_create_tensor_c(c_loc(array), 2, c_loc(shape), &
                                        TORCH_FLOAT64, dev, tensor%ptr)

        if (status /= 0) then
            write(*,'(A,I0)') 'Error creating tensor, status: ', status
            call print_last_error()
        end if
    end function torch_tensor_from_array_real64_2d

    function torch_tensor_from_array_real64_3d(array, device) result(tensor)
        real(c_double), target, intent(in) :: array(:,:,:)
        integer(torch_device), intent(in), optional :: device
        type(torch_tensor) :: tensor

        integer(c_int64_t) :: shape(3)
        integer(torch_device) :: dev
        integer(c_int) :: status

        shape(1) = size(array, 1)
        shape(2) = size(array, 2)
        shape(3) = size(array, 3)
        dev = TORCH_DEVICE_CPU
        if (present(device)) dev = device

        status = ftorch_create_tensor_c(c_loc(array), 3, c_loc(shape), &
                                        TORCH_FLOAT64, dev, tensor%ptr)

        if (status /= 0) then
            write(*,'(A,I0)') 'Error creating tensor, status: ', status
            call print_last_error()
        end if
    end function torch_tensor_from_array_real64_3d

    ! Tensor extraction to real32 arrays
    subroutine torch_tensor_to_array_real32_1d(tensor, array)
        type(torch_tensor), intent(in) :: tensor
        real(c_float), target, intent(out) :: array(:)
        integer(c_int) :: status

        status = ftorch_tensor_to_array_c(tensor%ptr, c_loc(array))

        if (status /= 0) then
            write(*,'(A,I0)') 'Error extracting tensor data, status: ', status
            call print_last_error()
        end if
    end subroutine torch_tensor_to_array_real32_1d

    subroutine torch_tensor_to_array_real32_2d(tensor, array)
        type(torch_tensor), intent(in) :: tensor
        real(c_float), target, intent(out) :: array(:,:)
        integer(c_int) :: status

        status = ftorch_tensor_to_array_c(tensor%ptr, c_loc(array))

        if (status /= 0) then
            write(*,'(A,I0)') 'Error extracting tensor data, status: ', status
            call print_last_error()
        end if
    end subroutine torch_tensor_to_array_real32_2d

    subroutine torch_tensor_to_array_real32_3d(tensor, array)
        type(torch_tensor), intent(in) :: tensor
        real(c_float), target, intent(out) :: array(:,:,:)
        integer(c_int) :: status

        status = ftorch_tensor_to_array_c(tensor%ptr, c_loc(array))

        if (status /= 0) then
            write(*,'(A,I0)') 'Error extracting tensor data, status: ', status
            call print_last_error()
        end if
    end subroutine torch_tensor_to_array_real32_3d

    ! Tensor extraction to real64 arrays
    subroutine torch_tensor_to_array_real64_1d(tensor, array)
        type(torch_tensor), intent(in) :: tensor
        real(c_double), target, intent(out) :: array(:)
        integer(c_int) :: status

        status = ftorch_tensor_to_array_c(tensor%ptr, c_loc(array))

        if (status /= 0) then
            write(*,'(A,I0)') 'Error extracting tensor data, status: ', status
            call print_last_error()
        end if
    end subroutine torch_tensor_to_array_real64_1d

    subroutine torch_tensor_to_array_real64_2d(tensor, array)
        type(torch_tensor), intent(in) :: tensor
        real(c_double), target, intent(out) :: array(:,:)
        integer(c_int) :: status

        status = ftorch_tensor_to_array_c(tensor%ptr, c_loc(array))

        if (status /= 0) then
            write(*,'(A,I0)') 'Error extracting tensor data, status: ', status
            call print_last_error()
        end if
    end subroutine torch_tensor_to_array_real64_2d

    subroutine torch_tensor_to_array_real64_3d(tensor, array)
        type(torch_tensor), intent(in) :: tensor
        real(c_double), target, intent(out) :: array(:,:,:)
        integer(c_int) :: status

        status = ftorch_tensor_to_array_c(tensor%ptr, c_loc(array))

        if (status /= 0) then
            write(*,'(A,I0)') 'Error extracting tensor data, status: ', status
            call print_last_error()
        end if
    end subroutine torch_tensor_to_array_real64_3d

end module ftorch

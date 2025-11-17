cmake_minimum_required(VERSION 3.18)
project(FortranTorch VERSION 1.0.0 LANGUAGES C CXX Fortran)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Fortran module output directory
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add include directory
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# Build the C++ library
add_library(fortran_torch_cpp SHARED
    src/cpp/fortran_torch.cpp
)

target_link_libraries(fortran_torch_cpp ${TORCH_LIBRARIES})

# Set RPATH for the C++ library
set_target_properties(fortran_torch_cpp PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Build the Fortran module
add_library(ftorch SHARED
    src/fortran/ftorch.f90
)

target_link_libraries(ftorch fortran_torch_cpp)

# Installation
install(TARGETS fortran_torch_cpp ftorch
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES include/fortran_torch.h
    DESTINATION include
)

install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/
    DESTINATION include/fortran
    FILES_MATCHING PATTERN "*.mod"
)

# Examples
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples/fortran)
endif()

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "FortranTorch Configuration Summary")
message(STATUS "===================================")
message(STATUS "  Version:           ${PROJECT_VERSION}")
message(STATUS "  Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Fortran compiler:  ${CMAKE_Fortran_COMPILER}")
message(STATUS "  C++ compiler:      ${CMAKE_CXX_COMPILER}")
message(STATUS "  PyTorch found:     ${Torch_DIR}")
message(STATUS "  CUDA available:    ${CUDA_AVAILABLE}")
message(STATUS "  Build examples:    ${BUILD_EXAMPLES}")
message(STATUS "  Build tests:       ${BUILD_TESTS}")
message(STATUS "")

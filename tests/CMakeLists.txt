# Tests CMakeLists.txt

# Enable testing
enable_testing()

# Test executables
add_executable(test_basic fortran/test_basic.f90)
target_link_libraries(test_basic ftorch fortran_torch_cpp ${TORCH_LIBRARIES})
target_include_directories(test_basic PRIVATE ${CMAKE_Fortran_MODULE_DIRECTORY})

add_executable(test_inference fortran/test_inference.f90)
target_link_libraries(test_inference ftorch fortran_torch_cpp ${TORCH_LIBRARIES})
target_include_directories(test_inference PRIVATE ${CMAKE_Fortran_MODULE_DIRECTORY})

# Add tests to CTest
add_test(NAME BasicTests COMMAND test_basic)

# Inference test requires model file
add_test(NAME InferenceTest COMMAND test_inference
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Set environment for tests
set_tests_properties(BasicTests PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${TORCH_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}"
)

set_tests_properties(InferenceTest PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${TORCH_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}"
)

# Copy Python test script to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/python/create_test_model.py
    ${CMAKE_CURRENT_BINARY_DIR}/create_test_model.py
    COPYONLY
)

# Custom target to create test model
add_custom_target(create_test_model
    COMMAND ${CMAKE_COMMAND} -E echo "Creating test model..."
    COMMAND python ${CMAKE_CURRENT_BINARY_DIR}/create_test_model.py
            ${CMAKE_CURRENT_BINARY_DIR}/test_model.pt
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating test model with Python"
)

# Custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_basic test_inference
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

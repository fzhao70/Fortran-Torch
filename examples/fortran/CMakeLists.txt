# Examples CMakeLists.txt

# Simple example
add_executable(simple_example simple_example.f90)
target_link_libraries(simple_example ftorch fortran_torch_cpp ${TORCH_LIBRARIES})
target_include_directories(simple_example PRIVATE ${CMAKE_Fortran_MODULE_DIRECTORY})

# Weather model example
add_executable(weather_model_example weather_model_example.f90)
target_link_libraries(weather_model_example ftorch fortran_torch_cpp ${TORCH_LIBRARIES})
target_include_directories(weather_model_example PRIVATE ${CMAKE_Fortran_MODULE_DIRECTORY})

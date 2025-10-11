cmake_minimum_required(VERSION 3.10)
project(BackendTrainer)

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(../include)

# Create executable
add_executable(backend_trainer backend_trainer.cpp)

# Link with the neural network library
target_link_libraries(backend_trainer nnlib)

# If OpenMP is found, use it for performance
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(backend_trainer OpenMP::OpenMP_CXX)
endif()
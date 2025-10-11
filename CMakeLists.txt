cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenMP)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h" "include/*.hpp")

# Define the library
add_library(nnlib ${SOURCES} ${HEADERS})

# Define executables
add_executable(main examples/main.cpp)
add_executable(xor_example examples/xor_example.cpp)
add_executable(mnist_example examples/mnist_example.cpp)
add_executable(mnist_example_improved examples/mnist_example_improved.cpp)

target_link_libraries(main nnlib)
target_link_libraries(xor_example nnlib)
target_link_libraries(mnist_example nnlib)
target_link_libraries(mnist_example_improved nnlib)

# If OpenMP is found, use it for performance
if(OpenMP_CXX_FOUND)
    target_link_libraries(nnlib OpenMP::OpenMP_CXX)
    target_link_libraries(main OpenMP::OpenMP_CXX)
    target_link_libraries(xor_example OpenMP::OpenMP_CXX)
    target_link_libraries(mnist_example OpenMP::OpenMP_CXX)
endif()

# Unit tests executable
add_executable(tests tests/test_main.cpp)
target_link_libraries(tests nnlib)

# Add backend trainer if available
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gui/backend_trainer.cpp")
    add_subdirectory(gui)
endif()
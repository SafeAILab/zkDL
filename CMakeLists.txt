cmake_minimum_required(VERSION 3.10)

project(zkDL)

# Specify CUDA path
set(CUDA_PATH "/usr/local/cuda" CACHE STRING "CUDA path")

# Find CUDA
list(APPEND CMAKE_PREFIX_PATH "${CUDA_PATH}")
find_package(CUDA REQUIRED)

# Find Torch
list(APPEND CMAKE_PREFIX_PATH "$ENV{VIRTUAL_ENV}/lib/python3.10/site-packages/torch")
find_package(Torch REQUIRED)

# Set NVCC flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_70)

# Specify the source files
file(GLOB SOURCES "*.cu")

# Add the executable
cuda_add_executable(${PROJECT_NAME} ${SOURCES})

# Link Torch libraries
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Link CUDA libraries
target_link_libraries(${PROJECT_NAME} "${CUDA_LIBRARIES}")

# If you have additional libraries or include directories, specify them here

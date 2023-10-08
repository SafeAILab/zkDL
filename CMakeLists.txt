cmake_minimum_required(VERSION 3.12)

project(zkDL LANGUAGES CXX CUDA)

# Setting CUDA architecture
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")
set(CMAKE_CUDA_STANDARD 14)

# Find Torch using the libtorch distribution
# list(APPEND CMAKE_PREFIX_PATH "${YOUR_LIBTORCH_PATH}")
find_package(Torch REQUIRED)

# Specify the source files
file(GLOB SOURCES "*.cu")

# Create an executable for each source file
foreach(SOURCE_FILE ${SOURCES})
    # Extract the file name without the extension
    get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)

    # Add the executable
    add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})

    # Link Torch libraries
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${TORCH_LIBRARIES})
    set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CXX_STANDARD 14)
endforeach()

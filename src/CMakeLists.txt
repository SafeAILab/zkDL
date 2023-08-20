# CMakeLists.txt in the src folder

# Find CUDA package
find_package(CUDA REQUIRED)

# Set CUDA architecture (change this to match your target architecture)
set(CUDA_ARCH "sm_70" CACHE STRING "CUDA architecture")

# Add CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${CUDA_ARCH}")

# Include paths
include_directories(${CMAKE_SOURCE_DIR}/bls12-381)

add_library(FrTensorLib fr-tensor.cu fr-tensor.cuh)

add_library(UtilsLib timer.hpp)
set_target_properties(UtilsLib PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(main main.cu)
target_link_libraries(main ECLib FrTensorLib UtilsLib)



# # Add source files
# set(SOURCES
#     main.cu
#     fr-tensor.cu
# )

# # Create an executable
# add_executable(main ${SOURCES})

# # Link CUDA libraries
# target_link_libraries(main CUDA::cudart)

# # Set C++ standard
# set_target_properties(main PROPERTIES
#     CXX_STANDARD 14
#     CXX_STANDARD_REQUIRED ON
# )

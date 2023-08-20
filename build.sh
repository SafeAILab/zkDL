#!/bin/bash

# Specify the paths
PROJECT_ROOT=$(pwd)
BUILD_DIR=${PROJECT_ROOT}/build

# Create the build directory if it doesn't exist
mkdir -p ${BUILD_DIR}

# Navigate to the build directory
cd ${BUILD_DIR}

# Run CMake to generate build files
cmake ${PROJECT_ROOT}

# Build the project using make (adjust this if using a different build system)
make

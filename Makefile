# Specify the CUDA installation path
CUDA_PATH ?= /usr/local/cuda

# Compilers
NVCC := $(CUDA_PATH)/bin/nvcc

# Include and library paths
INCLUDES := -I$(CUDA_PATH)/include
LIBS     := -L$(CUDA_PATH)/lib64 -lcudart

# NVCC compiler flags
NVCC_FLAGS := -arch=sm_70

# Source and target files
SRCS := $(wildcard *.cu)
TARGETS := $(SRCS:.cu=)

all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $< -o $@

clean:
	rm -f $(TARGETS)

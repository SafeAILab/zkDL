# Specify the CUDA and libtorch installation paths
CUDA_PATH ?= /usr/local/cuda
LIBTORCH_PATH ?= $(VIRTUAL_ENV)/lib/python3.10/site-packages/torch

# Compilers
NVCC := $(CUDA_PATH)/bin/nvcc

# Include and library paths
INCLUDES := -I$(CUDA_PATH)/include -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include
#LIBS     := -L$(CUDA_PATH)/lib64 -lcudart -L$(LIBTORCH_PATH)/lib -ltorch -lc10
LIBS := -L$(CUDA_PATH)/lib64 -lcudart -L$(LIBTORCH_PATH)/lib -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda


# NVCC compiler flags
NVCC_FLAGS := -arch=sm_70

# Source and target files
SRCS := $(wildcard *.cu)
TARGETS := $(SRCS:.cu=)

all: $(TARGETS)

%: %.cu
	$(NVCC)  $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $< -o $@ --linker-options=-rpath,$(LIBTORCH_PATH)/lib

clean:
	rm -f $(TARGETS)

# Specify the CUDA and libtorch installation paths
CUDA_PATH ?= /usr/local/cuda
LIBTORCH_PATH ?= $(VIRTUAL_ENV)/lib/python3.10/site-packages/torch

# Compilers
NVCC := $(CUDA_PATH)/bin/nvcc

# Include and library paths
INCLUDES := -I$(CUDA_PATH)/include -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include
LIBS := -L$(CUDA_PATH)/lib64 -lcudart -L$(LIBTORCH_PATH)/lib -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda

# NVCC compiler flags
# note: -D_GLIBCXX_USE_CXX11_ABI=0 may not be necessary and was only added due to some weird compilation errors with the torch library
# std=c++17 was also needed due to some errors with the torch library
NVCC_FLAGS := -arch=sm_86 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0

# Source and object files
CU_SRCS := $(wildcard *.cu)
CU_OBJS := $(CU_SRCS:.cu=.o)
CPP_SRCS := $(wildcard *.cpp)
CPP_OBJS := $(CPP_SRCS:.cpp=.o)
TARGET := demo

# note: --copy-dt-needed-entries and --no-as-needed were added to the linker options to fix some weird errors with the torch library
# -dlto is a flag to enable link time optimization
# -dc is a flag needed for separate compilation

all: $(TARGET)

$(TARGET): $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@ --linker-options=-rpath,$(LIBTORCH_PATH)/lib,--copy-dt-needed-entries,--no-as-needed -dlto

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc -dlto $< -o $@

%.o: %.cpp
	$(NVCC) -x cu $(NVCC_FLAGS) $(INCLUDES) -dc -dlto $< -o $@

clean:
	rm -f $(CU_OBJS) $(CPP_OBJS) $(TARGET)
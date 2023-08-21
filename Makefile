# Specify the CUDA installation path
CUDA_PATH ?= /usr/local/cuda

# Compilers
NVCC := $(CUDA_PATH)/bin/nvcc

# Include and library paths
INCLUDES := -I$(CUDA_PATH)/include
LIBS     := -L$(CUDA_PATH)/lib64 -lcudart

# NVCC compiler flags
NVCC_FLAGS := -arch=sm_70 # Adjust this for your GPU architecture

# Source and target files
SRC := main.cu
TARGET := my_program

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $< -o $@

clean:
	rm -f $(TARGET)

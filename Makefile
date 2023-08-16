# Makefile for CUDA Project

# Compiler and flags
NVCC = nvcc
CFLAGS = -std=c++14

# Source files
CU_FILES := $(wildcard *.cu)
OBJ_FILES := $(patsubst %.cu, %.o, $(CU_FILES))

# Target executable
TARGET = my_cuda_app

# CUDA architecture
CUDA_ARCH = -arch=sm_70  # Adjust to your target GPU architecture

# Make rules
all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) $^ -o $@

%.o: %.cu
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) -c $< -o $@

clean:
	rm -f $(OBJ_FILES) $(TARGET)

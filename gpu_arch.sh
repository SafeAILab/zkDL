#!/bin/bash

# Check if nvidia-smi works
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found, ensure your Nvidia GPU drivers are installed correctly."
    exit 1
fi

# calculate compute capability
COMPUTE_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)

if [ -z "$COMPUTE_CAPABILITY" ]; then
    echo "Failed to detect GPU Compute Capability"
    exit 1
fi

# get compute capability from retrieved value
ARCH="sm_$(echo $COMPUTE_CAPABILITY | tr -d '.')"

echo $ARCH
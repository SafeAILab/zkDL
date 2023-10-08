#!/bin/bash
#SBATCH --gpus=v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-00:59
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-hongyanz
#SBATCH --mail-user=haochen.sun@uwaterloo.ca
#SBATCH --mail-type=ALL

#!/bin/bash

# Load necessary modules
module load gcc cuda/11.4 cmake protobuf cudnn python/3.10

# Set up virtual environment and activate it
virtualenv --no-download --clear ~/ENV && source ~/ENV/bin/activate

# Install Python packages
pip install --no-index torch numpy

# Run model Python script
python model.py

# CMake build commands
cmake -B build -S . -DCMAKE_PREFIX_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath=$VIRTUAL_ENV/lib/python3.10/site-packages/torch/lib -DCMAKE_SKIP_RPATH=ON
cmake --build build

# Run the resulting binary
./build/load

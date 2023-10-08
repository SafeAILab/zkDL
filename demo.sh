#!/bin/bash
#SBATCH --gpus=v100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-00:59
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-hongyanz
#SBATCH --mail-user=haochen.sun@uwaterloo.ca
#SBATCH --mail-type=ALL

set -e  # Exit immediately if a command exits with a non-zero status

# Load necessary modules
module load gcc cuda/11.4 cmake protobuf cudnn python/3.10

# You can activate a virtual environment if you need to run the Python model
# But you shouldn't need it for compiling your CUDA project with CMake
virtualenv --no-download --clear ~/ENV && source ~/ENV/bin/activate

# If you need to run a Python model
pip install --no-index torch numpy
python model.py

# Build using CMake
cmake -B build -S . -DCMAKE_PREFIX_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath=$VIRTUAL_ENV/lib/python3.10/site-packages/torch/lib -DCMAKE_SKIP_RPATH=ON
cmake --build build --target demo-check

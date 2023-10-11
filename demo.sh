#!/bin/bash
#SBATCH --gpus=v100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-00:59
#SBATCH --output=%N-%j.out

set -e  # Exit immediately if a command exits with a non-zero status

# Load necessary modules
module load gcc cuda/11.4 cmake protobuf cudnn python/3.10

# You can activate a virtual environment if you need to run the Python model
# But you shouldn't need it for compiling your CUDA project with CMake
virtualenv --no-download --clear $SLURM_TMPDIR/ENV && source $SLURM_TMPDIR/ENV/bin/activate

# If you need to run a Python model
pip install --no-index torch numpy
python model.py

make clean
make demo

./demo traced_model.pt sample_input.pt

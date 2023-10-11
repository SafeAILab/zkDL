#!/bin/bash
#SBATCH --gpus=v100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-00:59
#SBATCH --output=%N-%j.out

set -e  # Exit immediately if a command exits with a non-zero status

# Load necessary modules (If you are using a different cluster, you may need to change this)
module load gcc cuda/11.4 cmake protobuf cudnn python/3.10

virtualenv --no-download --clear $SLURM_TMPDIR/ENV && source $SLURM_TMPDIR/ENV/bin/activate

pip install torch numpy
python model.py

make clean
make demo

./demo traced_model.pt sample_input.pt

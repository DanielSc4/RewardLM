#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=python_example
#SBATCH --mem=800


# single CPU only script
module purge
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.7.0

cd /home1/p313544
source .venv/bin/activate

echo "Python version: $(python --version)"

# User's vars
## All scripts must be in the PATH_TO_PRJ/ directory!
PATH_TO_PRJ=/home1/p313544/Documents/RewardLM
SCRIPT_NAME=job_test.py

echo "Executing python script..."
python $PATH_TO_PRJ/$SCRIPT_NAME




echo "Done!"
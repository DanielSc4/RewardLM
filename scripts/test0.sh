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

# User's vars
## All scripts must be in the PATH_TO_PRJ/scripts directory!
PATH_TO_PRJ=/home1/p313544/Documents/RewardLM
SCRIPT_NAME=job_test.py

# vars for execution
## colors
NC=$(tput sgr0)
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
BLUE=$(tput setaf 4)


cd /home1/p313544
source .venv/bin/activate

echo "${BLUE}Python version:${NC} $(python --version)"

echo "${BLUE}Executing python script...${NC}"
python $PATH_TO_PRJ/scripts/$SCRIPT_NAME

echo "${GREEN}Done!${NC}"
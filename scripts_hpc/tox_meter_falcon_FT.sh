#!/bin/bash
#SBATCH --job-name=tox_meter_falcon_FT
#SBATCH --time=7:00:00
#SBATCH --mem=30GB
#SBATCH --gpus-per-node=a100.20gb:1
#SBATCH --output=/home1/p313544/slurm_logs/%x.%j.out


# single CPU only script
module purge
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.7.0

cd /home1/p313544
source .venv/bin/activate

echo "Python version: $(python --version)"
nvidia-smi
pwd

# User's vars
## All scripts must be in the PATH_TO_PRJ/scripts directory!
PATH_TO_PRJ=/home1/p313544/Documents/RewardLM
SCRIPT_NAME=scriptToxicity.py


# checkpoint save path
export PATH_TO_STORAGE=/scratch/p313544/storage_cache/ret_toxicity


cd $PATH_TO_PRJ
echo "Executing 3 python script..."
# echo "[PT]"
# python $SCRIPT_NAME -c falcon7B
echo "[FT]"
python $SCRIPT_NAME -c falcon7b-FT
# echo "[RL]"
# python $SCRIPT_NAME -c falcon7b-RL


echo "Done!"

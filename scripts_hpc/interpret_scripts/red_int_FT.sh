#!/bin/bash
#SBATCH --job-name=red_int_FT
#SBATCH --time=02:20:00
#SBATCH --mem=70GB
#SBATCH --gpus-per-node=a100:1
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
SCRIPT_NAME=interpretability/gen_attributes.py


# checkpoint save path
export PATH_TO_STORAGE=/scratch/p313544/storage_cache/interpret_models/


cd $PATH_TO_PRJ

# echo "[PT]"
# python $SCRIPT_NAME -m configs/RedPajama-INCITE-Chat-3B-v1.yaml -i interpretability/interp_configs/i_debug_prod.yaml
echo "[FT]"
python $SCRIPT_NAME -m configs/RedPajama-INCITE-Chat-3B-LoRA-FT.yaml -i interpretability/interp_configs/i_debug_prod.yaml -s 4800
# echo "[RL]"
# python $SCRIPT_NAME -m configs/RedPajama-INCITE-Chat-3B-LoRA-RL.yaml -i interpretability/interp_configs/i_debug_prod.yaml



echo "Done!"

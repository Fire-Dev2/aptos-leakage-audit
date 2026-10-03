#!/bin/bash
#SBATCH --job-name=nnmb
#SBATCH --account=YOUR_PROJECT
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=36:00:00
#SBATCH --output=nnmb_%j.log
# nnMobileNet reruns: 6 runs, 3 per A100. If the job ends early, `sbatch nnmb.sbatch` again - it resumes.
cd "$HOME/leakage"
module load miniconda3/24.1.2-py310
source activate diffmic
nvidia-smi
python run_nnmb.py

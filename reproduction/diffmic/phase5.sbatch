#!/bin/bash
#SBATCH --job-name=phase5_diffmic
#SBATCH --account=YOUR_PROJECT
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=48:00:00
#SBATCH --output=phase5_%j.log
# 4 DiffMIC runs (R0 1234, R1 1234, R1 1, R1 2), two per A100. If the job ends before all finish,
# just `sbatch phase5.sbatch` again - finished runs are skipped, unfinished ones resume.
cd "$HOME/leakage"
module load miniconda3/24.1.2-py310
source activate diffmic
nvidia-smi
python run_phase5.py
